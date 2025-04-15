# Quoridor AI
## Modul: "Neue Konzepte KI"

Valentin Czekalla
Samuel Hilpert
Jared Heinrich

## Repository Structure
The Repository is structured in different rust crates.
There are several crates functioning as libraries and two crates producing an
executable.

The existing Crates are:
- quoridor:
Library providing an api to manage the state of the game.
- matrix:
Library providing a matrix struct and different operations on it, the main operation being multiplication of two matrices.
- neural_network:
Library providing a neural network struct and operations on it, the main
operation being a feed forward operation over the neural network.
- neural_network_logger:
Library to serialize and deserialize a neural network into json format.
- render:
Library providing functions to help rendering the game.
- evolution_training:
Executable that starts the learing process based on a random generation.
Configuration can be made in the main.rs file of this crate.
The neural networks of different generations are logged in the file
evolution_history.json.
The executable can be run by executing the command `cargo run --release` within the directory of the crate.
- play_ai_game:
Executable that allows to view two neural networks against each other. To specify which neural networks should play 
the line `logger::read_specific_lines(&[1, 2], "../evolution_training/evolution_history.json")` in main.rs can be modified.
To let the ai play the next move press "Enter".

