use acyclib::trainer::logger::ansi;

#[derive(Clone, Copy)]
pub struct TestDataset<'a> {
    /// Path to test dataset.
    pub path: &'a str,
    /// Frequency of validation loss (run validation every `freq` superbatches).
    pub freq: usize,
    /// Number of batches to consume from the validation loader each evaluation.
    pub batches_per_eval: Option<usize>,
}

impl<'a> TestDataset<'a> {
    pub fn at(path: &'a str) -> TestDataset<'a> {
        Self { path, freq: 1, batches_per_eval: None }
    }
}

pub struct LocalSettings<'a> {
    /// Number of threads to make available for training, in addition
    /// to the main trainer thread (used only for loading data if training
    /// with GPU).
    pub threads: usize,
    /// Path to a test dataset, will calculate vaidation loss over this dataset.
    pub test_set: Option<TestDataset<'a>>,
    /// Directory to write checkpoints to.
    pub output_directory: &'a str,
    /// Number of batches that the dataloader can prepare and put in a queue before
    /// they are processed in training.
    pub batch_queue_size: usize,
}

impl LocalSettings<'_> {
    pub fn display(&self) {
        println!("Threads                : {}", ansi(self.threads, 31));
        println!("Output Path            : {}", ansi(self.output_directory, "32;1"));
        if let Some(test_set) = self.test_set {
            println!("Validation Interval    : {} superbatch(es)", ansi(test_set.freq, 31));
            if let Some(batches) = test_set.batches_per_eval {
                println!("Validation Batches     : {}", ansi(batches, 31));
            }
        }
    }
}
