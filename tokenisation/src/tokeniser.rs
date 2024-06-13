use std::collections::HashMap;
use std::collections::HashSet;
pub struct Tokeniser {
    pub char_to_index: HashMap<char, usize>,
    pub index_to_char: HashMap<usize, char>,
    vocab_size: usize,
}

impl Tokeniser {
    pub fn new(text: &str) -> Tokeniser {
        // find unique characters in text
        let mut unique_chars = HashSet::new();
        for c in text.chars() {
            unique_chars.insert(c);
        }
        // Build character to index Hash map
        let mut char_to_index: HashMap<char, usize> = HashMap::new();
        let mut index = 0;
        for c in unique_chars {
            char_to_index.insert(c, index);
            index += 1;
        }
        // Build index to character hash map
        let mut index_to_char: HashMap<usize, char> = HashMap::new();
        for (c, idx) in &char_to_index {
            index_to_char.insert(*idx, *c);
        }

        Tokeniser {
            vocab_size: char_to_index.len(),
            char_to_index,
            index_to_char,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut result = Vec::new();
        for c in text.chars() {
            let idx = self
                .char_to_index
                .get(&c)
                .expect("character not found in tokeniser!");
            result.push(*idx);
        }
        result
    }

    pub fn decode(&self, indices: &[usize]) -> String {
        let mut result = String::new();
        for idx in indices {
            let char = self.index_to_char.get(&idx).unwrap();
            result.push(*char);
        }
        result
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tokeniser_encode() {
        let text = "The quick brown dog.";
        let tokeniser = Tokeniser::new(text);
        let tokens = tokeniser.encode("wow The brow on The dog");

        println!("{:?}", tokens);
    }

    #[test]
    fn tokeniser_decode() {
        let text = "The quick brown dog.";
        let tokeniser = Tokeniser::new(text);

        let letters = tokeniser.decode(&vec![3, 0, 0, 2]);

        println!("{:?}", letters);
    }
}
