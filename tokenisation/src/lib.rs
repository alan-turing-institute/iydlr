//! The text I am linking below is Louisa May Alcotts Little Women and it will serve as our dataset to run the model on
//! tokenisation] (.../mytext.txt)
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::read_to_string;
use std::{char, fs};
pub mod batch_generator;
pub mod tokeniser;

pub fn read_txt(path: &str) -> String {
    fs::read_to_string(path).unwrap()
}
pub fn hashmap_basic(char_map: HashMap<char, usize>) -> Vec<(char, usize)> {
    let mut char_freq: Vec<_> = char_map.into_iter().collect();
    char_freq.sort_by(|a, b| b.1.cmp(&a.1));
    char_freq
}
pub fn main() {
    //println!("{}", story);
    //println!("First 1000 characters: {}", first_thou);
    let story: String = read_txt("my_text.txt");
    let punc_chars = [
        '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<',
        '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ',
    ];
    let first_thou: String = story.chars().take(1000).collect();
    // println!("{:?}", story.chars().take(10).collect::<Vec<_>>());

    println!("Number of characters: {}", story.chars().count());

    let file_path = "my_text.txt";
    let var_name = match read_to_string(file_path) {
        Ok(content) => content,
        Err(error) => {
            eprintln!("Failed to read file: {}", error);
            return;
        }
    };
    let text = var_name;
    let mut unique_chars = HashSet::new();
    for c in text.chars() {
        unique_chars.insert(c);
    }
    // println!("Unique Characters:");
    // for c in &unique_chars {
    //     println!("{}", c);
    // }
    //character map which maps characters to frequency number (i.e. how often the characters appear in our story text)
    let mut char_map = HashMap::new();
    for c in story.chars() {
        char_map
            .entry(c)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }
    print!("Number of unique characters: {}", char_map.len());
    let char_freq: Vec<(char, usize)> = hashmap_basic(char_map.clone());
    println!("Unique characters from most to least commonly occurring:");
    for (char, freq) in char_freq.clone() {
        println!(" {} : {}", char, freq);
    }

    //Character to Index Hashmap, which maps characters to index numbers
    let mut char_to_index: HashMap<char, i32> = HashMap::new();
    let mut index: i32 = 0;
    for (char, _) in char_freq {
        char_to_index.insert(char, index);
        index += 1;
    }
    println!("Character to Index HashMap:");
    char_to_index.iter().for_each(|(char, index)| {
        println!("Character: {}, Index: {}", char, index);
    });
    //Index to Character Hashmap, which maps index numbers to characters starting from 0
    println!("Index to Character HashMap:");
    let mut index_to_char: HashMap<i32, char> = HashMap::new();
    for (char, idx) in &char_to_index {
        index_to_char.insert(*idx, *char);
    }
    let mut indices: Vec<_> = index_to_char.keys().collect();
    indices.sort();
    for index in indices {
        if let Some(char) = index_to_char.get(index) {
            println!("Index: {}, Character: {}", index, char);
        }
    }
}
// println!("Index to Character HashMap:");
// index_to_char.iter().for_each(|(index, char)| {
//     println!("Index: {}, Character: {}", index, char);
// });

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_world() {
        main()
    }
}
