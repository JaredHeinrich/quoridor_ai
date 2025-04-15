# Quoridor AI  
## Module: *New Concepts in AI*

**Authors:**  
Valentin Czekalla  
Samuel Hilpert  
Jared Heinrich  

---

## Repository Structure

The repository is organized into multiple Rust crates.  
Several crates function as libraries, and two crates produce executables.

### Existing Crates

- **`quoridor`**  
    A library that provides an API to manage the game state of Quoridor.

- **`matrix`**  
    A library that implements a matrix structure and various matrix operations.  
    The main functionality is the multiplication of two matrices.

- **`neural_network`**  
    A library that defines a neural network structure and associated operations.  
    Its core feature is the feedforward operation through the network.

- **`neural_network_logger`**  
    A library for serializing and deserializing neural networks in JSON format.

- **`render`**  
    A library that provides helper functions for rendering the game.

- **`evolution_training`** *(executable)*  
    Starts the learning process based on random generation.  
    Configuration can be modified in the `main.rs` file of this crate.  
    Neural networks of different generations are logged in the file `evolution_history.json`.  
    To run the executable, use the following command inside the crate directory:  
    ```bash
    cargo run --release
    ```

- **`play_ai_game`** *(executable)*
    Allows viewing a match between two neural networks.
    To specify which networks should play, modify the following line in main.rs:
    ```rust
    logger::read_specific_lines(&[1, 2], "../evolution_training/evolution_history.json")
    ```
    Where the path specifys which file should be used to load the neural
    networks, and the lines specify in which line of the file the neural networks
    can be found. To let the AI make the next move press "Enter".
