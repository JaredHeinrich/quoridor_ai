use anyhow::Result;
use neural_network_logger::models::LogEntry;
use rand::Rng;

use crate::evolution::agent::Agent; // Add this import
use crate::evolution::generation::Generation;
use crate::settings::Settings;
use neural_network_logger::logger::read_specific_lines;

/// Creates a new generation based on the results of the previous generation
///
/// This function applies the evolutionary selection process:
/// 1. Top-performing agents survive based on the survival rate
/// 2. Some agents are randomly reactivated from the top performers
/// 3. The rest of the population is filled with mutated copies of survivors
///
/// For the first generation (when old_generation is None), it creates an entirely new population.
///
/// # Arguments
/// * `old_generation` - The previous generation, or None if this is the first generation
/// * `settings` - The settings controlling the evolution process
/// * `generation_number` - The current generation number (0-indexed)
///
/// # Returns
/// A new Generation ready for evaluation
pub fn select_next_generation(
    old_generation: Option<&mut Generation>,
    settings: &Settings,
) -> Result<Generation> {
    // If it's the first generation or no old generation is provided, create a new one
    if old_generation.is_none() {
        return Generation::create_initial(settings);
    }

    let old_generation = old_generation.unwrap();
    old_generation.sort_by_fitness()?;

    // Calculate the number of survivors and reactivators
    let survivor_count = settings.survivor_count();
    let reactivator_count = settings.reactivator_count();

    // Create a new generation
    let mut new_generation = Generation {
        agents: Vec::with_capacity(settings.generation_size),
        generation_index: old_generation.generation_index + 1,
        mutation_rate: old_generation.mutation_rate
            * (1.0 - settings.mutation_rate_decrease as f64).max(0.0),
    };

    // Add the survivors (top performers) to the new generation
    for i in 0..survivor_count {
        let mut survivor = old_generation.agents[i].clone();
        survivor.reset_fitness();
        new_generation.agents.push(survivor);
    }

    // Add the reactivators (randomly selected from all of the previous generations)
    reactivate_random_agents(reactivator_count, &mut new_generation, settings, None)?;

    let remaining_count = settings.generation_size - new_generation.agents.len();

    // Fill the rest with mutated neural networks
    if remaining_count > 0 {
        let current_mutation_rate = new_generation.mutation_rate;

        // If the number remaining equals survivor_count, each survivor is mutated once
        if remaining_count == survivor_count {
            // Mutate each survivor once
            for i in 0..remaining_count {
                let mut mutated_agent = old_generation.agents[i].clone();
                mutated_agent.neural_network.mutate(current_mutation_rate);
                mutated_agent.reset_fitness();
                new_generation.agents.push(mutated_agent);
            }
        } else {
            // Otherwise, randomly select agents from survivors to mutate
            let mut rng = rand::rng();
            for _ in 0..remaining_count {
                let random_index = rng.random_range(0..survivor_count);
                let mut mutated_agent = old_generation.agents[random_index].clone();
                mutated_agent.neural_network.mutate(current_mutation_rate);
                mutated_agent.reset_fitness();
                new_generation.agents.push(mutated_agent);
            }
        }
    }

    Ok(new_generation)
}

/// Reactivates a number of agents from all of the previous generations
fn reactivate_random_agents(
    reactivator_count: usize,
    new_generation: &mut Generation,
    settings: &Settings,
    // optional function parameter with same signature as read_specific_lines for mocking in tests
    reader_fn: Option<fn(&[usize], &str) -> Result<Vec<LogEntry>>>,
) -> Result<()> {
    if new_generation.generation_index == 0 || reactivator_count == 0 {
        return Ok(());
    }

    let mut reload_lines_list: Vec<usize> = Vec::with_capacity(reactivator_count);
    let mut rng = rand::rng();
    let max_line_index = new_generation.generation_index * settings.generation_size;

    for _ in 0..reactivator_count {
        let line_index = rng.random_range(0..max_line_index);
        reload_lines_list.push(line_index);
    }

    // Use provided function or default to real implementation
    let reader = reader_fn.unwrap_or(read_specific_lines);
    let logentries = reader(&reload_lines_list, &settings.log_file)?;

    for line in logentries {
        let agent = Agent::new(line.neural_network);
        new_generation.agents.push(agent);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_network::neural_network::NeuralNetwork;
    use neural_network_logger::models::LogEntry;

    // Mock implementation for testing
    fn mock_read_specific_lines(line_count: &[usize], _file_path: &str) -> Result<Vec<LogEntry>> {
        // Create predefined neural networks for testing
        let mut mock_entries = Vec::new();

        for i in 0..line_count.len() {
            let nn = NeuralNetwork::new(&vec![10, 10, 5]).unwrap();
            mock_entries.push(LogEntry {
                neural_network: nn,
                generation_index: 0,
                placement: i,
                fitness: Some(0.0),
            });
        }

        Ok(mock_entries)
    }

    // Helper function to create a generation with specific fitness values
    fn create_test_generation(settings: &Settings, fitness_values: Vec<f64>) -> Result<Generation> {
        let mut generation = Generation::create_initial(settings)?;

        // Assign fitness values
        for (i, fitness) in fitness_values.into_iter().enumerate() {
            if i < generation.agents.len() {
                generation.agents[i].set_fitness(fitness);
            }
        }

        Ok(generation)
    }

    #[test]
    fn test_first_generation() -> Result<()> {
        let settings = Settings::default();
        let generation = select_next_generation(None, &settings)?;

        // Check if the correct number of agents was created
        assert_eq!(generation.size(), settings.generation_size);

        // Check that all agents have zero fitness
        for agent in &generation.agents {
            assert_eq!(agent.get_fitness(), 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_selection_preserves_top_performers() -> Result<()> {
        let settings = Settings::default()
            .with_generation_size(10)
            .with_survival_rate(0.5)
            .with_reactivation_rate(0.0);

        // Create a generation with descending fitness values (9, 8, 7, ...)
        let fitness_values: Vec<f64> = (0..10).rev().map(|v| v as f64).collect();
        let mut old_gen = create_test_generation(&settings, fitness_values)?;

        // Store original top performer NNs before selection
        let top_nn_weights: Vec<_> = old_gen
            .agents
            .iter()
            .take(settings.survivor_count())
            .map(|a| a.neural_network.weights.clone())
            .collect();

        let new_gen = select_next_generation(Some(&mut old_gen), &settings)?;

        // Verify survivors have the same neural networks as the top performers
        for i in 0..settings.survivor_count() {
            assert_eq!(new_gen.agents[i].neural_network.weights, top_nn_weights[i]);
        }

        Ok(())
    }

    #[test]
    fn test_reactivation_functionality() -> Result<()> {
        let settings = Settings::default()
            .with_generation_size(10)
            .with_survival_rate(0.3) // 3 survivors
            .with_reactivation_rate(0.4); // 4 reactivators

        let mut old_gen = create_test_generation(&settings, vec![5.0; 10])?;
        old_gen.generation_index = 5; // Set a non-zero generation index

        // Create a new generation directly to test with mocked function
        let mut new_gen = Generation {
            agents: Vec::with_capacity(settings.generation_size),
            generation_index: old_gen.generation_index + 1,
            mutation_rate: old_gen.mutation_rate,
        };

        // Add the survivors first
        for i in 0..settings.survivor_count() {
            let mut survivor = old_gen.agents[i].clone();
            survivor.reset_fitness();
            new_gen.agents.push(survivor);
        }

        // Test the reactivation with mocked function
        reactivate_random_agents(
            settings.reactivator_count(),
            &mut new_gen,
            &settings,
            Some(mock_read_specific_lines),
        )?;

        // Check that the correct number of agents were reactivated
        let reactivated_count = new_gen
            .agents
            .iter()
            .filter(|agent| agent.neural_network.layer_sizes == vec![10, 10, 5])
            .count();
        assert_eq!(
            reactivated_count,
            settings.reactivator_count(),
            "Expected {} agents to be reactivated but found {}",
            settings.reactivator_count(),
            reactivated_count
        );

        Ok(())
    }

    #[test]
    fn test_mutation_rate_decrease() -> Result<()> {
        let initial_rate = 0.5;
        let decrease = 0.2; // 20% decrease

        let settings = Settings::default()
            .with_mutation_rate(initial_rate)
            .with_mutation_rate_decrease(decrease)
            .with_reactivation_rate(0.0);

        let mut old_gen = create_test_generation(&settings, vec![1.0; 10])?;
        old_gen.mutation_rate = initial_rate;

        let new_gen = select_next_generation(Some(&mut old_gen), &settings)?;

        // Expected rate after 20% decrease
        let expected_rate = initial_rate * (1.0 - decrease as f64);
        assert_eq!(new_gen.mutation_rate, expected_rate);

        Ok(())
    }
}
