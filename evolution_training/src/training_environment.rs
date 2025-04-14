use anyhow::Result;
use plotters::prelude::*;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::evolution::generation::Generation;
use crate::evolution::selection::select_next_generation;
use crate::game_adapter::board_encoder::{distance_to_goal, encode_board};
use crate::game_adapter::move_decoder::decode_move;
use crate::game_adapter::reward::{reward_simple_per_game, reward_symmetric_per_game};
use crate::settings::Settings;
use neural_network::neural_network::{NeuralNetwork, OutputActivation};
use neural_network_logger::logger::{log_generation, log_single_log_entry};
use neural_network_logger::models::LogEntry;
use quoridor::game_state::Game;

#[derive(Debug, Clone, PartialEq)]
pub enum GameResult {
    Win(usize),
    Invalid,
    Draw,
}

/// Manages the training environment for neural network agents playing Quoridor
pub struct TrainingEnvironment {
    /// Configuration settings for the training process
    pub settings: Settings,
    /// The current generation of agents
    pub current_generation: Generation,
    /// The current generation number (0-indexed)
    pub generation_number: usize,
    /// Fitness history for plotting (generation, max_fitness, avg_fitness, min_fitness)
    pub fitness_history: Vec<(usize, f64, f64, f64)>,
    /// Generation time history for plotting (generation, time_in_seconds)
    pub generation_time_history: Vec<(usize, f64)>,
}

impl TrainingEnvironment {
    /// Create a new training environment with the given settings
    pub fn new(settings: Settings) -> Self {
        let current_generation = Generation::create_initial(&settings).unwrap();
        TrainingEnvironment {
            settings,
            current_generation,
            generation_number: 0,
            fitness_history: Vec::new(),
            generation_time_history: Vec::new(),
        }
    }

    /// Initialize the training environment by creating the first generation
    pub fn reset(&mut self) -> Result<()> {
        self.generation_number = 0;
        self.current_generation = Generation::create_initial(&self.settings)?;
        self.fitness_history.clear();
        self.generation_time_history.clear();
        println!(
            "Initialized generation 0 with {} agents",
            self.settings.generation_size
        );
        Ok(())
    }

    /// Run the complete evolutionary training process for the specified number of generations
    pub fn run_evolution(&mut self) -> Result<()> {
        self.reset()?;

        println!(
            "Starting evolutionary training for {} generations",
            self.settings.number_of_generations
        );
        let start_time = Instant::now();

        // Log the initial generation
        let generation = &mut self.current_generation;
        log_generation(generation, &self.settings.log_file)?;

        for current_generation_num in 0..self.settings.number_of_generations {
            self.generation_number = current_generation_num;
            println!("─────────────────────────────────────────────────");
            println!(
                "Starting generation {}/{}",
                self.generation_number, self.settings.number_of_generations
            );

            let gen_start_time = Instant::now();

            // Play all games in this generation and update fitness scores
            self.play_generation()?;

            // Log and display results
            self.log_generation_results()?;

            let generation_duration = gen_start_time.elapsed();

            // Store generation time for plotting (in seconds)
            self.generation_time_history
                .push((self.generation_number, generation_duration.as_secs_f64()));

            println!(
                "Generation {} completed in {:.2?}",
                self.generation_number, generation_duration
            );

            // Create next generation (except for the last iteration)
            if self.generation_number + 1 < self.settings.number_of_generations {
                self.evolve_population()?;
            }
        }

        let total_duration = start_time.elapsed();
        println!(
            "Training complete after {} generations in {:.2?}",
            self.generation_number + 1,
            total_duration
        );
        println!("Results saved to {}", self.settings.log_file);

        // Plot fitness history
        self.plot_fitness_history()?;

        // Plot generation times
        self.plot_generation_times()?;

        Ok(())
    }

    /// Log and display the results of the current generation
    fn log_generation_results(&mut self) -> Result<()> {
        let generation = &mut self.current_generation;
        // Sort agents by fitness
        generation.sort_by_fitness()?;

        // Log generation to file
        log_generation(generation, &self.settings.log_file)?;

        // Display stats
        let top_fitness = generation.agents[0].get_fitness();
        let avg_fitness: f64 = generation
            .agents
            .iter()
            .map(|a| a.get_fitness())
            .sum::<f64>()
            / generation.agents.len() as f64;
        let min_fitness = generation.agents.last().unwrap().get_fitness();

        println!("Generation {} results:", self.generation_number + 1);
        println!("  Top fitness: {:.2}", top_fitness);
        println!("  Avg fitness: {:.2}", avg_fitness);
        println!("  Min fitness: {:.2}", min_fitness);

        // Update fitness history
        self.fitness_history.push((
            self.generation_number,
            top_fitness,
            avg_fitness,
            min_fitness,
        ));

        Ok(())
    }

    /// Play all agents in a generation against each other
    pub fn play_generation(&mut self) -> Result<()> {
        // Take ownership of the generation temporarily using std::mem::take
        let generation = std::mem::take(&mut self.current_generation);
        let generation_size = generation.agents.len();

        // Create shared, thread-safe reference to the generation
        let generation_arc = Arc::new(Mutex::new(generation));

        // Create all pairs of agents that will play against each other
        let agent_pairs: Vec<(usize, usize)> = (0..generation_size)
            .flat_map(|i| (i + 1..generation_size).map(move |j| (i, j)))
            .collect();

        let num_games = agent_pairs.len();
        println!(
            "Playing {} games in generation {}",
            num_games,
            self.generation_number + 1
        );

        // Track progress for user feedback
        let progress = Arc::new(Mutex::new((0, num_games)));

        // Play games in parallel
        agent_pairs.par_iter().for_each(|(i, j)| {
            if let Err(e) = self.play_game_between_agents(*i, *j, Arc::clone(&generation_arc)) {
                eprintln!("Error playing game between agents {} and {}: {}", i, j, e);
            }

            // Update and display progress
            let mut progress_guard = progress.lock().unwrap();
            progress_guard.0 += 1;
            if progress_guard.0 % (num_games / 20).max(1) == 0 || progress_guard.0 == num_games {
                println!(
                    "  Progress: {}/{} games ({:.1}%)",
                    progress_guard.0,
                    progress_guard.1,
                    100.0 * progress_guard.0 as f64 / progress_guard.1 as f64
                );
            }
        });

        // Get generation back from Arc<Mutex<>>
        let generation = Arc::try_unwrap(generation_arc)
            .expect("Failed to unwrap Arc")
            .into_inner()
            .expect("Failed to unwrap Mutex");

        self.current_generation = generation;
        Ok(())
    }

    /// Play a single game between two agents and update their fitness scores
    fn play_game_between_agents(
        &self,
        agent0_index: usize,
        agent1_index: usize,
        generation_arc: Arc<Mutex<Generation>>,
    ) -> Result<()> {
        // Get references to the agent neural networks
        let agent0_nn;
        let agent1_nn;

        {
            let generation = generation_arc.lock().unwrap();
            agent0_nn = generation
                .get_neural_network(agent0_index)
                .ok_or_else(|| anyhow::anyhow!("Agent index {} out of bounds", agent0_index))?
                .clone();
            agent1_nn = generation
                .get_neural_network(agent1_index)
                .ok_or_else(|| anyhow::anyhow!("Agent index {} out of bounds", agent1_index))?
                .clone();
        }

        // Play the game and get rewards
        let (reward0, reward1) = self.play_single_game(&agent0_nn, &agent1_nn)?;

        // Update agent fitness
        {
            let mut generation = generation_arc.lock().unwrap();

            if let Some(agent1) = generation.agents.get_mut(agent0_index) {
                agent1.increase_fitness(reward0);
            }

            if let Some(agent2) = generation.agents.get_mut(agent1_index) {
                agent2.increase_fitness(reward1);
            }
        }

        Ok(())
    }

    /// Play a single game between two neural networks and return rewards
    fn play_single_game(
        &self,
        neural_network0: &NeuralNetwork,
        neural_network1: &NeuralNetwork,
    ) -> Result<(f64, f64)> {
        // Create a new game
        let mut game = Game::new(
            self.settings.board_size as i16,
            self.settings.walls_per_player as i16,
        );

        let output_activation = if self.settings.play_deterministically {
            OutputActivation::Sigmoid
        } else {
            OutputActivation::Softmax
        };

        let mut moves_played = self.settings.max_moves_per_player;
        // Play until someone wins or max moves reached
        for move_counter in 0..self.settings.max_moves_per_player * 2 {
            // Determine which agent's turn it is
            let current_player_index = game.current_pawn;
            let current_agent_nn = if current_player_index == 0 {
                neural_network0
            } else {
                neural_network1
            };
            let action: GameResult =
                self.nn_move(current_agent_nn, &mut game, move_counter, output_activation);
            // If the game is won, break out of the loop
            if let GameResult::Win(moves_to_win) = action {
                moves_played = moves_to_win;
                break;
            } else if let GameResult::Invalid = action {
                let log_entry = LogEntry {
                    generation_index: usize::MAX,
                    placement: usize::MAX,
                    neural_network: neural_network0.clone(),
                    fitness: None,
                };
                let _result = log_single_log_entry(&log_entry, &self.settings.log_file);
                let log_entry = LogEntry {
                    generation_index: usize::MAX,
                    placement: current_player_index, //use current_player_index to identify who had invalid moves
                    neural_network: neural_network1.clone(),
                    fitness: None,
                };
                let _result = log_single_log_entry(&log_entry, &self.settings.log_file);

                break;
            }
        }
        // Calculate rewards for both players
        self.calculate_rewards(&game, moves_played)
    }

    pub fn nn_move(
        &self,
        neural_network: &NeuralNetwork,
        game: &mut Game,
        move_counter: usize,
        output_activation: OutputActivation,
    ) -> GameResult {
        // Get and execute move
        let game_state = encode_board(&game);
        let nn_output = neural_network.feed_forward(game_state.unwrap(), output_activation);
        let game_move = decode_move(&nn_output.unwrap(), &game, &self.settings);

        // Execute move
        if let Err(_) = &game.make_move(game_move.unwrap()) {
            // If move execution failed, we'll end the game and consider it a draw
            // This shouldn't happen if move_decoder is working correctly, but just in case
            print!("Invalid move executed, move_decoder malfunctioning, ending game.");
            return GameResult::Invalid;
        }

        // Check if game is over// Check for win condition or max moves
        if self.is_win(&game) {
            let moves_played = (move_counter + 1) / 2;
            return GameResult::Win(moves_played);
        } else {
            return GameResult::Draw;
        }
    }

    /// Check if the game is over (win or max moves reached)
    fn is_win(&self, game: &Game) -> bool {
        // Check if any player has reached their goal line
        let player0_distance = distance_to_goal(game, 0);
        let player1_distance = distance_to_goal(game, 1);

        player0_distance == 0 || player1_distance == 0
    }

    /// Calculate rewards for both players based on game outcome and settings
    fn calculate_rewards(&self, game: &Game, moves_played: usize) -> Result<(f64, f64)> {
        match self.settings.reward_function {
            crate::game_adapter::reward::RewardFunction::Simple => Ok((
                reward_simple_per_game(game, 0, moves_played, &self.settings),
                reward_simple_per_game(game, 1, moves_played, &self.settings),
            )),
            crate::game_adapter::reward::RewardFunction::Symmetric => Ok((
                reward_symmetric_per_game(game, 0, moves_played, &self.settings),
                reward_symmetric_per_game(game, 1, moves_played, &self.settings),
            )),
        }
    }

    /// Evolve the population to create the next generation
    pub fn evolve_population(&mut self) -> Result<()> {
        println!(
            "Evolving population for generation {}",
            self.generation_number + 1
        );

        // Create next generation using selection strategy
        let next_generation =
            select_next_generation(Some(&mut self.current_generation), &self.settings)?;

        self.current_generation = next_generation;
        self.generation_number += 1;

        // Report survival and mutation rates
        let survivors = self.settings.survivor_count();
        let reactivated = self.settings.reactivator_count();

        println!(
            "Created generation {} with {} agents:",
            self.generation_number,
            &self.current_generation.agents.len()
        );
        println!(
            "  Survivors: {} ({:.1}%)",
            survivors,
            100.0 * survivors as f64 / self.settings.generation_size as f64
        );
        println!(
            "  Reactivated: {} ({:.1}%)",
            reactivated,
            100.0 * reactivated as f64 / self.settings.generation_size as f64
        );
        println!(
            "  Mutation rate: {:.3}",
            &self.current_generation.mutation_rate
        );

        Ok(())
    }

    pub fn plot_fitness_history(&self) -> Result<()> {
        if self.fitness_history.is_empty() {
            println!("No fitness data to plot");
            return Ok(());
        }

        let output_file = format!(
            "{}_fitness_plot.png",
            self.settings.log_file.replace(".json", "")
        );
        println!("Generating fitness plot at: {}", output_file);

        // Create the plot
        let root = BitMapBackend::new(&output_file, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        // Find min and max values for y-axis
        let min_y = self
            .fitness_history
            .iter()
            .map(|(_gen, _max, _avg, min)| *min)
            .fold(f64::INFINITY, |a, b| a.min(b))
            .min(0.0);

        let max_y = self
            .fitness_history
            .iter()
            .map(|(_gen, max, _avg, _min)| *max)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b))
            * 1.1; // Add 10% margin

        let max_gen = self.fitness_history.len() as u32;

        let mut chart = ChartBuilder::on(&root)
            .caption("Fitness over Generations", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0u32..max_gen, min_y..max_y)?;

        chart
            .configure_mesh()
            .x_desc("Generation")
            .y_desc("Fitness")
            .axis_desc_style(("sans-serif", 15))
            .draw()?;

        // Plot max fitness
        chart
            .draw_series(LineSeries::new(
                self.fitness_history
                    .iter()
                    .map(|(gen, max, _avg, _min)| (*gen as u32, *max)),
                &RED,
            ))?
            .label("Max Fitness")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        // Plot average fitness
        chart
            .draw_series(LineSeries::new(
                self.fitness_history
                    .iter()
                    .map(|(gen, _max, avg, _min)| (*gen as u32, *avg)),
                &GREEN,
            ))?
            .label("Avg Fitness")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

        // Plot min fitness
        chart
            .draw_series(LineSeries::new(
                self.fitness_history
                    .iter()
                    .map(|(gen, _max, _avg, min)| (*gen as u32, *min)),
                &BLUE,
            ))?
            .label("Min Fitness")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        println!("Fitness plot generated at: {}", output_file);

        Ok(())
    }

    pub fn plot_generation_times(&self) -> Result<()> {
        if self.generation_time_history.is_empty() {
            println!("No generation time data to plot");
            return Ok(());
        }

        let output_file = format!(
            "{}_time_plot.png",
            self.settings.log_file.replace(".json", "")
        );
        println!("Generating generation time plot at: {}", output_file);

        // Create the plot
        let root = BitMapBackend::new(&output_file, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        // Find min and max values for y-axis
        let min_y = 0.0; // Time can't be negative
        let max_y = self
            .generation_time_history
            .iter()
            .map(|(_gen, time)| *time)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b))
            * 1.1; // Add 10% margin

        let max_gen = self.generation_time_history.len() as u32;

        let mut chart = ChartBuilder::on(&root)
            .caption("Generation Time", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0u32..max_gen, min_y..max_y)?;

        chart
            .configure_mesh()
            .x_desc("Generation")
            .y_desc("Time (seconds)")
            .axis_desc_style(("sans-serif", 15))
            .draw()?;

        // Plot generation time
        chart
            .draw_series(LineSeries::new(
                self.generation_time_history
                    .iter()
                    .map(|(gen, time)| (*gen as u32, *time)),
                &BLUE,
            ))?
            .label("Generation Time (s)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        // Add a trend line (moving average) if we have enough data points
        if self.generation_time_history.len() >= 3 {
            // Calculate moving average (window size of 3)
            let mut trend_data = Vec::new();
            for i in 1..self.generation_time_history.len() - 1 {
                let avg_time = (self.generation_time_history[i - 1].1
                    + self.generation_time_history[i].1
                    + self.generation_time_history[i + 1].1)
                    / 3.0;
                trend_data.push((self.generation_time_history[i].0 as u32, avg_time));
            }

            chart
                .draw_series(LineSeries::new(trend_data, &RED.mix(0.5)))?
                .label("Moving Average (3)")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED.mix(0.5)));
        }

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        println!("Generation time plot generated at: {}", output_file);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quoridor::vector::Vector;

    #[test]
    fn test_initialization() -> Result<()> {
        // Create settings with small generation size for testing
        let settings = Settings::default().with_generation_size(10);

        let mut env = TrainingEnvironment::new(settings);

        // Test initialization
        env.reset()?;
        assert_eq!(env.current_generation.agents.len(), 10);
        assert_eq!(env.generation_number, 0);

        Ok(())
    }

    #[test]
    fn test_is_win() -> Result<()> {
        let settings = Settings::default().with_max_moves_per_player(50);
        let env = TrainingEnvironment::new(settings);

        // New game, not won
        let game = Game::new(9, 10);
        assert!(!env.is_win(&game));

        // Create a game with player 0 at goal line
        let mut win_game = Game::new(9, 10);
        win_game.pawns[0].position = Vector::new(4, 0); // Goal line is at y=0
        assert!(env.is_win(&win_game));

        Ok(())
    }

    #[test]
    fn test_calculate_rewards() -> Result<()> {
        let settings = Settings::default().with_reward_coefficients(
            crate::game_adapter::reward::RewardFunction::Simple,
            100.0, // win reward
            -5.0,  // own distance punishment
            2.0,   // other distance reward
            1.0,   // per saved turn reward
        );

        let env = TrainingEnvironment::new(settings);

        // Create a custom game with player 0 winning
        let mut game = Game::new(9, 10);
        game.pawns[0].position = Vector::new(4, 0); // At goal line
        game.pawns[1].position = Vector::new(4, 6); // Some distance from goal

        // Calculate rewards
        let (reward0, reward1) = env.calculate_rewards(&game, 20)?;

        // Player 0 won and should get win reward + saved turns + distance rewards
        assert!(reward0 > 0.0);
        // Check specific reward components
        let expected_reward0 = 100.0 + (1.0 * (50.0 - 20.0)) + (-5.0 * 0.0) + (2.0 * 2.0);
        assert_eq!(reward0, expected_reward0);

        // Player 1 lost and should get only distance-based rewards
        let expected_reward1 = 0.0 + (-5.0 * 2.0) + (2.0 * 0.0);
        assert_eq!(reward1, expected_reward1);

        Ok(())
    }

    #[test]
    fn test_evolve_population() -> Result<()> {
        let settings = Settings::default()
            .with_generation_size(10)
            .with_survival_rate(0.5)
            .with_reactivation_rate(0.0);

        let mut env = TrainingEnvironment::new(settings);
        env.reset()?;

        assert_eq!(env.generation_number, 0);

        // Set some fitness values to ensure deterministic sorting
        let generation = &mut env.current_generation;
        for (i, agent) in generation.agents.iter_mut().enumerate() {
            agent.set_fitness(i as f64);
        }

        // Evolve to next generation
        env.evolve_population()?;

        assert_eq!(env.generation_number, 1);
        assert_eq!(env.current_generation.agents.len(), 10);

        Ok(())
    }

    #[test]
    fn test_play_single_game() -> Result<()> {
        let settings = Settings::default().with_max_moves_per_player(10);
        let env = TrainingEnvironment::new(settings);

        // Create two neural networks
        let neural_network0 = NeuralNetwork::new(&vec![
            env.settings.input_layer_size,
            100,
            env.settings.output_layer_size,
        ])?;
        let neural_network1 = NeuralNetwork::new(&vec![
            env.settings.input_layer_size,
            100,
            env.settings.output_layer_size,
        ])?;

        // Play a game and get rewards
        let (reward0, reward1) = env.play_single_game(&neural_network0, &neural_network1)?;

        // Basic sanity checks
        assert!(reward0.is_finite());
        assert!(reward1.is_finite());

        Ok(())
    }
}
