use acyclib::trainer::{
    DataLoadingError,
    dataloader::{DataLoader, HostDenseMatrix, HostMatrix, HostSparseMatrix, PreparedBatchHost},
    schedule::TrainingSteps,
};

use crate::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    trainer::schedule::wdl::WdlScheduler,
    value::loader::{self, DenseInput, PreparedData, SparseInput},
};

pub struct ValueDataLoader<I, O, D, W>
where
    I: SparseInputType,
    I::RequiredDataType: loader::LoadableDataType,
    O: OutputBuckets<I::RequiredDataType>,
    D: loader::DataLoader<I::RequiredDataType>,
{
    pub dataloader: loader::DefaultDataLoader<I, O, D>,
    pub steps: TrainingSteps,
    pub threads: usize,
    pub wdl: W,
}

impl<I, O, D, W> DataLoader for ValueDataLoader<I, O, D, W>
where
    I: SparseInputType,
    I::RequiredDataType: loader::LoadableDataType,
    O: OutputBuckets<I::RequiredDataType>,
    W: WdlScheduler,
    D: loader::DataLoader<I::RequiredDataType>,
{
    type Error = DataLoadingError;

    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(
        self,
        batch_size: usize,
        mut f: F,
    ) -> Result<(), DataLoadingError> {
        let ValueDataLoader { dataloader, steps, threads, wdl } = self;
        let start_batch = steps.batches_per_superbatch * (steps.start_superbatch - 1);

        assert_eq!(batch_size, steps.batch_size);

        let mut batch_no = 0;
        let mut superbatch = 1;

        dataloader.load_and_map_batches(start_batch, batch_size, |batch| {
            let blend = wdl.blend(batch_no, superbatch, steps.end_superbatch);
            let prepared_data = dataloader.prepare(batch, threads, blend);

            batch_no += 1;

            if batch_no % steps.batches_per_superbatch == 0 {
                batch_no = 0;
                superbatch += 1;
            }

            f(prepared_data.into())
        });

        Ok(())
    }
}

#[derive(Clone)]
pub struct ValidationDataLoader<I, O, D, W>
where
    I: SparseInputType,
    I::RequiredDataType: loader::LoadableDataType,
    O: OutputBuckets<I::RequiredDataType>,
    D: loader::DataLoader<I::RequiredDataType>,
    W: WdlScheduler,
{
    dataloader: loader::DefaultDataLoader<I, O, D>,
    steps: TrainingSteps,
    threads: usize,
    wdl: W,
    max_superbatch: usize,
}

impl<I, O, D, W> ValidationDataLoader<I, O, D, W>
where
    I: SparseInputType,
    I::RequiredDataType: loader::LoadableDataType,
    O: OutputBuckets<I::RequiredDataType>,
    D: loader::DataLoader<I::RequiredDataType>,
    W: WdlScheduler,
{
    pub fn new(
        dataloader: loader::DefaultDataLoader<I, O, D>,
        steps: TrainingSteps,
        threads: usize,
        wdl: W,
        max_superbatch: usize,
    ) -> Self {
        let mut steps = steps;
        if steps.batches_per_superbatch == 0 {
            steps.batches_per_superbatch = 1;
        }

        Self { dataloader, steps, threads, wdl, max_superbatch: max_superbatch.max(1) }
    }

    pub fn batches_per_evaluation(&self) -> usize {
        self.steps.batches_per_superbatch.max(1)
    }

    pub fn batches_per_cycle(&self) -> usize {
        let span = self.steps.end_superbatch.saturating_sub(self.steps.start_superbatch).saturating_add(1).max(1);
        self.batches_per_evaluation().saturating_mul(span)
    }

    fn start_offset_batches(&self) -> usize {
        self.batches_per_evaluation().saturating_mul(self.steps.start_superbatch.saturating_sub(1))
    }

    pub fn map_prepared_batches<F>(&self, start_batch: usize, current_superbatch: usize, mut f: F)
    where
        F: FnMut(usize, PreparedBatchHost) -> bool,
    {
        if self.steps.batch_size == 0 {
            return;
        }

        let cycle_batches = self.batches_per_cycle().max(1);
        let start_offset = self.start_offset_batches();
        let start_batch = start_batch % cycle_batches;
        let start_batch = start_offset + start_batch;

        let mut batches_produced = 0usize;
        let mut batch_index = 0usize;
        let mut should_break = false;

        self.dataloader.load_and_map_batches(start_batch, self.steps.batch_size, |batch| {
            if should_break || batches_produced >= self.steps.batches_per_superbatch {
                return true;
            }

            let blend = self.wdl.blend(batch_index, current_superbatch, self.max_superbatch);
            let prepared = self.dataloader.prepare(batch, self.threads, blend);

            batches_produced += 1;
            batch_index += 1;
            should_break = f(batches_produced, prepared.into());
            should_break || batches_produced >= self.steps.batches_per_superbatch
        });
    }
}

impl<I: SparseInputType, O> From<PreparedData<I, O>> for PreparedBatchHost {
    fn from(prepared_data: PreparedData<I, O>) -> Self {
        let batch_size = prepared_data.batch_size;

        let mut host_data = PreparedBatchHost { batch_size, inputs: Default::default() };

        unsafe {
            let SparseInput { value, max_active, shape } = prepared_data.stm;
            let stm = HostSparseMatrix::new(value, Some(batch_size), shape, max_active);
            let _ = host_data.inputs.insert("stm".to_string(), HostMatrix::Sparse(stm));

            let SparseInput { value, max_active, shape } = prepared_data.nstm;
            let ntm = HostSparseMatrix::new(value, Some(batch_size), shape, max_active);
            let _ = host_data.inputs.insert("nstm".to_string(), HostMatrix::Sparse(ntm));

            let SparseInput { value, max_active, shape } = prepared_data.buckets;
            let buckets = HostSparseMatrix::new(value, Some(batch_size), shape, max_active);
            let _ = host_data.inputs.insert("buckets".to_string(), HostMatrix::Sparse(buckets));
        }

        let DenseInput { value, shape } = prepared_data.targets;
        let targets = HostDenseMatrix::new(value, Some(batch_size), shape);
        let _ = host_data.inputs.insert("targets".to_string(), HostMatrix::Dense(targets));

        let DenseInput { value, shape } = prepared_data.weights;
        let weights = HostDenseMatrix::new(value, Some(batch_size), shape);
        let _ = host_data.inputs.insert("entry_weights".to_string(), HostMatrix::Dense(weights));

        host_data
    }
}
