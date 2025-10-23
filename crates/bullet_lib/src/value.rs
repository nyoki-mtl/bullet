pub(crate) mod builder;
mod dataloader;
pub mod loader;
mod save;

use std::cell::RefCell;

pub use builder::{NoOutputBuckets, ValueTrainerBuilder};

use acyclib::{
    graph::Node,
    trainer::{self, Trainer, logger, optimiser::OptimiserState},
};

use acyclib::{
    graph::{GraphNodeId, GraphNodeIdTy, like::GraphLike, save::SavedFormat},
    trainer::dataloader::{PreparedBatchDevice, PreparedBatchHost},
};

use crate::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    nn::{ExecutionContext, Graph},
    trainer::{
        schedule::{TrainingSchedule, lr::LrScheduler, wdl::WdlScheduler},
        settings::LocalSettings,
    },
    value::{
        dataloader::{ValidationDataLoader, ValueDataLoader},
        loader::{DefaultDataLoader, LoadableDataType},
    },
};

use crate::value::loader::PreparedData;

/// Value network trainer, generally for training NNUE networks.
pub struct ValueTrainer<
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
>(Trainer<ExecutionContext, Graph, Opt, ValueTrainerState<Inp, Out>>);

impl<Opt, Inp, Out> std::ops::Deref for ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    type Target = Trainer<ExecutionContext, Graph, Opt, ValueTrainerState<Inp, Out>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Opt, Inp, Out> std::ops::DerefMut for ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

type B<I> = fn(&<I as SparseInputType>::RequiredDataType, f32) -> f32;
type Wgt<I> = fn(&<I as SparseInputType>::RequiredDataType) -> f32;

pub struct ValueTrainerState<Inp: SparseInputType, Out> {
    input_getter: Inp,
    output_getter: Out,
    blend_getter: B<Inp>,
    weight_getter: Option<Wgt<Inp>>,
    _output_node: Node,
    saved_format: Vec<SavedFormat>,
    use_win_rate_model: bool,
    wdl: bool,
}

struct ValidationReport {
    average_loss: f32,
    batches: usize,
    positions: usize,
}

#[derive(Clone, Copy)]
pub(super) enum LossRecordKind {
    Train,
    Validation,
}

impl LossRecordKind {
    fn label(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct LossRecord {
    superbatch: usize,
    step_in_superbatch: usize,
    loss: f32,
    kind: LossRecordKind,
}

impl LossRecord {
    fn train(superbatch: usize, step_in_superbatch: usize, loss: f32) -> Self {
        Self { superbatch, step_in_superbatch, loss, kind: LossRecordKind::Train }
    }

    fn validation(superbatch: usize, step_in_superbatch: usize, loss: f32) -> Self {
        Self { superbatch, step_in_superbatch, loss, kind: LossRecordKind::Validation }
    }

    fn kind(&self) -> LossRecordKind {
        self.kind
    }

    fn superbatch(&self) -> usize {
        self.superbatch
    }

    fn step(&self) -> usize {
        self.step_in_superbatch
    }

    fn loss(&self) -> f32 {
        self.loss
    }
}

type CoreTrainer<Opt, Inp, Out> = Trainer<ExecutionContext, Graph, Opt, ValueTrainerState<Inp, Out>>;

struct ValidationContext<I, O, D, W>
where
    I: SparseInputType,
    I::RequiredDataType: LoadableDataType,
    O: OutputBuckets<I::RequiredDataType>,
    D: loader::DataLoader<I::RequiredDataType>,
    W: WdlScheduler,
{
    loader: ValidationDataLoader<I, O, D, W>,
    frequency_superbatch: usize,
    superbatches_until_eval: usize,
    device_batch: RefCell<Option<PreparedBatchDevice<ExecutionContext>>>,
    next_start_batch: usize,
}

impl<I, O, D, W> ValidationContext<I, O, D, W>
where
    I: SparseInputType,
    I::RequiredDataType: LoadableDataType,
    O: OutputBuckets<I::RequiredDataType>,
    D: loader::DataLoader<I::RequiredDataType>,
    W: WdlScheduler,
{
    fn new(loader: ValidationDataLoader<I, O, D, W>, frequency_superbatch: usize) -> Self {
        let (frequency_superbatch, superbatches_until_eval) =
            if frequency_superbatch == 0 { (0, 0) } else { (frequency_superbatch, frequency_superbatch) };

        Self {
            loader,
            frequency_superbatch,
            superbatches_until_eval,
            device_batch: RefCell::new(None),
            next_start_batch: 0,
        }
    }

    fn on_superbatch_end<Opt>(
        &mut self,
        trainer: &mut CoreTrainer<Opt, I, O>,
        superbatch: usize,
        record: &RefCell<Vec<LossRecord>>,
        batches_per_superbatch: usize,
    ) where
        Opt: OptimiserState<ExecutionContext>,
    {
        if self.frequency_superbatch == 0 {
            return;
        }

        if self.superbatches_until_eval > 0 {
            self.superbatches_until_eval -= 1;
        }

        if self.superbatches_until_eval == 0 {
            match self.run_validation(trainer, superbatch) {
                Ok(report) => {
                    if report.batches > 0 {
                        let colour = logger::num_cs();
                        println!(
                            "validation superbatch {} | loss {} | batches {} | positions {}",
                            logger::ansi(superbatch, colour),
                            logger::ansi(format!("{:.6}", report.average_loss), 31),
                            logger::ansi(report.batches, colour),
                            logger::ansi(report.positions, colour),
                        );
                        record.borrow_mut().push(LossRecord::validation(
                            superbatch,
                            batches_per_superbatch,
                            report.average_loss,
                        ));
                    } else {
                        println!(
                            "{}",
                            logger::ansi(
                                "Validation dataloader produced no batches; skipping loss report for this interval.",
                                "33"
                            )
                        );
                    }
                }
                Err(err) => {
                    panic!("Validation run failed: {err:?}");
                }
            }

            self.superbatches_until_eval = self.frequency_superbatch;
        }
    }

    fn run_validation<Opt>(
        &mut self,
        trainer: &mut CoreTrainer<Opt, I, O>,
        superbatch: usize,
    ) -> Result<ValidationReport, trainer::TrainerError<ExecutionContext>>
    where
        Opt: OptimiserState<ExecutionContext>,
    {
        let start_batch = self.next_start_batch;
        let mut total_loss = 0.0f32;
        let mut total_batches = 0usize;
        let mut total_positions = 0usize;
        let mut error: Option<trainer::TrainerError<ExecutionContext>> = None;

        self.loader.map_prepared_batches(start_batch, superbatch, |_, prepared| {
            if error.is_some() {
                return true;
            }

            let batch_size = prepared.batch_size.max(1);

            match self.evaluate_batch(trainer, prepared) {
                Ok(loss) => {
                    total_loss += loss;
                    total_batches += 1;
                    total_positions += batch_size;
                    false
                }
                Err(err) => {
                    error = Some(err);
                    true
                }
            }
        });

        if let Some(err) = error {
            return Err(err);
        }

        if total_batches == 0 {
            return Ok(ValidationReport { average_loss: 0.0, batches: 0, positions: 0 });
        }

        self.advance_batches(total_batches);

        Ok(ValidationReport {
            average_loss: total_loss / total_batches as f32,
            batches: total_batches,
            positions: total_positions,
        })
    }

    fn evaluate_batch<Opt>(
        &self,
        trainer: &mut CoreTrainer<Opt, I, O>,
        batch: PreparedBatchHost,
    ) -> Result<f32, trainer::TrainerError<ExecutionContext>>
    where
        Opt: OptimiserState<ExecutionContext>,
    {
        let batch_size = batch.batch_size.max(1);
        let graph = &mut trainer.optimiser.graph;

        let mut device_slot = self.device_batch.borrow_mut();
        if let Some(device_batch) = device_slot.as_mut() {
            device_batch.load_new_data(&batch).map_err(trainer::TrainerError::Unexpected)?;
        } else {
            let device_batch = PreparedBatchDevice::new(graph.devices(), &batch)
                .map_err(|err| trainer::TrainerError::Unexpected(err.into()))?;
            *device_slot = Some(device_batch);
        }

        let device_batch = device_slot.as_mut().expect("device batch should be initialised");
        device_batch.load_into_graph(graph)?;

        graph.execute_fn("zero_grads").map_err(trainer::TrainerError::Unexpected)?;
        graph.execute_fn("forward").map_err(trainer::TrainerError::Unexpected)?;

        let loss_sum = graph.get_output_value().map_err(trainer::TrainerError::Unexpected)?;

        Ok(loss_sum / batch_size as f32)
    }

    fn advance_batches(&mut self, batches: usize) {
        if batches == 0 {
            return;
        }

        let cycle = self.loader.batches_per_cycle().max(1);
        self.next_start_batch = (self.next_start_batch + batches) % cycle;
    }
}

impl<Opt, Inp, Out> ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Inp::RequiredDataType: LoadableDataType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn run<ValLoader>(
        &mut self,
        schedule: &TrainingSchedule<impl LrScheduler, impl WdlScheduler>,
        settings: &LocalSettings,
        dataloader: &impl loader::DataLoader<Inp::RequiredDataType>,
        validation_loader: Option<&ValLoader>,
    ) where
        ValLoader: loader::DataLoader<Inp::RequiredDataType> + Clone,
    {
        logger::clear_colours();
        println!("{}", logger::ansi("Training Preamble", "34;1"));

        schedule.display();
        settings.display();

        if let Some(test_set) = settings.test_set {
            println!("Validation Dataset     : {}", logger::ansi(test_set.path, "32;1"));
            println!("Validation Interval    : {} superbatch(es)", logger::ansi(test_set.freq, 31));
            if let Some(batches) = test_set.batches_per_eval {
                println!("Validation Batches     : {}", logger::ansi(batches, 31));
            }
        }

        let training_loader = DefaultDataLoader::new(
            self.state.input_getter.clone(),
            self.state.output_getter,
            self.state.blend_getter,
            self.state.weight_getter,
            self.state.use_win_rate_model,
            self.state.wdl,
            schedule.eval_scale,
            dataloader.clone(),
        );

        let steps = schedule.steps;

        let mut validation_context = match (settings.test_set, validation_loader) {
            (Some(test_set), Some(validation_loader)) => {
                if test_set.freq == 0 {
                    println!(
                        "{}",
                        logger::ansi(
                            "Validation dataset configured with evaluation frequency 0; skipping validation.",
                            "33"
                        )
                    );
                    None
                } else {
                    let validation_loader = DefaultDataLoader::new(
                        self.state.input_getter.clone(),
                        self.state.output_getter,
                        self.state.blend_getter,
                        self.state.weight_getter,
                        self.state.use_win_rate_model,
                        self.state.wdl,
                        schedule.eval_scale,
                        validation_loader.clone(),
                    );

                    let superbatch_frequency = test_set.freq;
                    let batches_per_eval = test_set
                        .batches_per_eval
                        .unwrap_or_else(|| (steps.batches_per_superbatch / superbatch_frequency.max(1)).max(1));
                    let mut validation_steps = steps;
                    validation_steps.batches_per_superbatch = batches_per_eval.max(1);
                    let max_superbatch = schedule.steps.end_superbatch.max(1);

                    Some(ValidationContext::new(
                        ValidationDataLoader::new(
                            validation_loader,
                            validation_steps,
                            settings.threads,
                            schedule.wdl_scheduler.clone(),
                            max_superbatch,
                        ),
                        superbatch_frequency,
                    ))
                }
            }
            (Some(_), None) => {
                println!(
                    "{}",
                    logger::ansi(
                        "Validation dataset configured but no validation loader supplied; skipping validation.",
                        "33"
                    )
                );
                None
            }
            _ => None,
        };

        let _ = std::fs::create_dir(settings.output_directory);

        let lr_scheduler = schedule.lr_scheduler.clone();

        let error_record = RefCell::new(Vec::<LossRecord>::new());
        let mut prev32_loss = 0.0;
        let mut prev32_count = 0usize;

        self.train_custom(
            trainer::schedule::TrainingSchedule {
                steps,
                log_rate: 128,
                lr_schedule: Box::new(|a, b| lr_scheduler.lr(a, b)),
            },
            ValueDataLoader {
                steps,
                threads: settings.threads,
                dataloader: training_loader,
                wdl: schedule.wdl_scheduler.clone(),
            },
            |_, superbatch, curr_batch, error| {
                prev32_loss += error;
                prev32_count += 1;

                if curr_batch % 32 == 0 || curr_batch == steps.batches_per_superbatch {
                    let divisor = prev32_count.max(1) as f32;
                    let averaged_loss = prev32_loss / divisor;
                    error_record.borrow_mut().push(LossRecord::train(superbatch, curr_batch, averaged_loss));
                    prev32_loss = 0.0;
                    prev32_count = 0;
                }
            },
            |trainer, superbatch| {
                if let Some(validation) = validation_context.as_mut() {
                    validation.on_superbatch_end(trainer, superbatch, &error_record, steps.batches_per_superbatch);
                }
                if superbatch % schedule.save_rate == 0 || superbatch == steps.end_superbatch {
                    let name = format!("{}-{superbatch}", schedule.net_id);
                    let path = format!("{}/{name}", settings.output_directory);
                    std::fs::create_dir(path.as_str()).unwrap_or(());
                    save::save_to_checkpoint(trainer, &path);
                    save::write_losses(&format!("{path}/log.txt"), &error_record.borrow());

                    println!("Saved [{}]", logger::ansi(name, 31));
                }
            },
        )
        .unwrap();
    }

    pub fn eval_raw_output(&mut self, fen: &str) -> Vec<f32>
    where
        Inp::RequiredDataType: std::str::FromStr<Err: std::fmt::Debug> + LoadableDataType,
    {
        let pos = format!("{fen} | 0 | 0.0").parse::<Inp::RequiredDataType>().unwrap();

        let prepared = PreparedData::new(
            self.state.input_getter.clone(),
            self.state.output_getter,
            self.state.blend_getter,
            self.state.weight_getter,
            self.state.use_win_rate_model,
            self.state.wdl,
            &[pos],
            1,
            1.0,
            1.0,
        );

        let host_data = PreparedBatchHost::from(prepared);

        let id = GraphNodeId::new(self.state._output_node.idx(), GraphNodeIdTy::Values);

        #[cfg(not(any(feature = "multigpu", feature = "cpu")))]
        let graph = &mut self.optimiser.graph;

        #[cfg(any(feature = "multigpu", feature = "cpu"))]
        let graph = self.optimiser.graph.primary_mut();

        let mut device_data = PreparedBatchDevice::new(graph.devices(), &host_data).unwrap();

        device_data.load_into_graph(graph).unwrap();

        graph.synchronise().unwrap();
        graph.forward().unwrap();

        let eval = graph.get(id).unwrap();

        let dense_vals = eval.dense();
        let mut vals = vec![0.0; dense_vals.size()];
        dense_vals.write_to_slice(&mut vals).unwrap();
        vals
    }

    pub fn eval(&mut self, fen: &str) -> f32
    where
        Inp::RequiredDataType: std::str::FromStr<Err: std::fmt::Debug> + LoadableDataType,
    {
        let vals = self.eval_raw_output(fen);

        match vals[..] {
            [mut loss, mut draw, mut win] => {
                let max = win.max(draw).max(loss);
                win = (win - max).exp();
                draw = (draw - max).exp();
                loss = (loss - max).exp();

                (win + draw / 2.0) / (win + draw + loss)
            }
            [score] => score,
            _ => panic!("Invalid output size!"),
        }
    }
}
