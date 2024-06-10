## Getting started

- Install Rust! Follow instructions
  [here](https://www.rust-lang.org/tools/install).
- For **vscode** users:
  - Install the `rust-analyzer` extension.
  - Open User Settings (JSON) with `CMD + SHIFT + P` the type
    `Open User Settings`.
  - Add these settings:
    ```json
    "[rust]": {
      "editor.defaultFormatter": "rust-lang.rust-analyzer",
      "editor.formatOnSave": true,
      "editor.rulers": [102]
    },
    ```

## How to use the repo

### Making branches and issues

- Create an issue and note the issue number (eg. `5`).
- Create a branch named `5-mybranch-name`.

### Create a new Rust Crate in the Rust Workspace (./iydlr/)

- `cd /path/to/iydlr`
- Decide if it's a library crate or a binary crate.
  - binary crate: `cargo new my-crate-name`
  - library crate: `cargo new my-crate-name --lib`

### Interfaces and how to use them

The goal is the create a Rust "thing" (could be a `struct` or an `enum`) that
_implements_ the interface. Which looks like this in Rust:

```rust
trait MyInterface {
  /// A method that takes a reference to `self` and returns `usize`.
  fn some_func(&self) -> usize;
}

struct MyStruct {
  some_field: usize,
}

impl MyInterface for MyStruct {
  fn some_func(&self) -> usize {
    // do some special logic
    *self.some_field + 2
  }
}
```

You will now be able to find that function implemented on the `struct` when you
instantiate the `struct`:

```rust
fn demo() -> () {
  let my_struct = MyStruct { some_field: 4 };

  let my_result = &my_struct.some_func();

  // `my_result` will be = 6
}
```

## Resources

### Rust resources

- The amazing [Rust Language Book](https://doc.rust-lang.org/book/): better
  writing than most novels, very clear and helpful.
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/): The
  hand-in-hand example based companion to the Rust Lang Book.

### Tokenisation resources

- [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)
  has a brilliant intro to tokenisation with a tutorial in tokenising the
  complete works of Shakespear with "character-level" tokenisation.
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE&t=2584s)
  is another video by Andrej Karpathy with a tutorial on more complex
  tokenisation approaches.

### Autograd resources

- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=7543s)
  is a very understanable and powerful approach to automatic differentiation
  that was a big motivation for this project.

### Transformer resources

- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4840s)
  really Oscar worthy stuff - throw together by Karpathy in a brief career pause
  between Tesla and OpenAI.
