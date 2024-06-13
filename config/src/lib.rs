#[derive(Debug)]
pub struct Config {
    pub batch_size: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub embed_dim: usize,
    pub num_head: usize,
    pub seed: u64,
    pub num_blocks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
}
