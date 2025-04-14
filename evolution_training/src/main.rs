pub mod error;
pub mod evolution;
pub mod game_adapter;
pub mod logging;
pub mod settings;
pub mod training_environment;

use std::process;

use crate::game_adapter::reward::RewardFunction;
use crate::settings::Settings;
use crate::training_environment::TrainingEnvironment;

fn run() -> anyhow::Result<()> {
    // Create and configure settings with reasonable defaults
    let settings = Settings::default()
        .with_generation_size(10)
        .with_network_architecture(vec![147, 128, 128, 128, 132])
        .with_survival_rate(0.4)
        .with_reactivation_rate(0.2)
        .with_mutation_rate(0.1)
        .with_mutation_rate_decrease(0.001)
        .with_reward_coefficients(
            RewardFunction::Symmetric,
            100.0,            // win reward
            -10.0,             // own distance punishment
            5.0,               // other distance reward
            2.0,               // per saved turn reward
        )
        .with_max_moves_per_player(50)
        .with_deterministic_play(true)
        .with_generation_count(100);
    
    // Validate settings
    settings.validate()?;
    
    println!("Starting Quoridor AI evolutionary training");
    println!("Configured with:");
    println!("  Population size: {}", settings.generation_size);
    println!("  Neural network: {:?}", settings.neural_network_layer_structure);
    println!("  Generations: {}", settings.number_of_generations);
    println!("  Survival rate: {:.2}", settings.survival_rate);
    println!("  Mutation rate: {:.2} (decrease: {:.3})", settings.mutation_rate, settings.mutation_rate_decrease);
    
    // Create and run the training environment
    let mut environment = TrainingEnvironment::new(settings);
    environment.run_evolution()?;
    
    println!("Training completed successfully!");
    
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error during training: {}", e);
        process::exit(1);
    }
}
