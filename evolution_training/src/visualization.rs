use crate::training_environment::TrainingEnvironment;
use anyhow::Result;
use neural_network::neural_network::NeuralNetwork;
use plotters::prelude::*;
use plotters::style::full_palette::{ORANGE, PURPLE}; // Updated to import ORANGE and PURPLE

/// Plot fitness metrics throughout the evolutionary process
pub fn plot_fitness_history(environment: &TrainingEnvironment) -> Result<()> {
    if environment.fitness_history.is_empty() {
        println!("No fitness data to plot");
        return Ok(());
    }

    let output_file = format!(
        "{}_fitness_plot.png",
        environment.settings.log_file.replace(".json", "")
    );
    println!("Generating fitness plot at: {}", output_file);

    // Create the plot
    let root = BitMapBackend::new(&output_file, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find min and max values for y-axis
    let min_y = environment
        .fitness_history
        .iter()
        .map(|(_gen, _max, _avg, min)| *min)
        .fold(f64::INFINITY, |a, b| a.min(b))
        .min(0.0);

    let max_y = environment
        .fitness_history
        .iter()
        .map(|(_gen, max, _avg, _min)| *max)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b))
        * 1.1; // Add 10% margin

    let max_gen = environment.fitness_history.len() as u32;

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
            environment
                .fitness_history
                .iter()
                .map(|(gen, max, _avg, _min)| (*gen as u32, *max)),
            &RED,
        ))?
        .label("Max Fitness")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Plot average fitness
    chart
        .draw_series(LineSeries::new(
            environment
                .fitness_history
                .iter()
                .map(|(gen, _max, avg, _min)| (*gen as u32, *avg)),
            &GREEN,
        ))?
        .label("Avg Fitness")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    // Plot min fitness
    chart
        .draw_series(LineSeries::new(
            environment
                .fitness_history
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

/// Plot generation execution times
pub fn plot_generation_times(environment: &TrainingEnvironment) -> Result<()> {
    if environment.generation_time_history.is_empty() {
        println!("No generation time data to plot");
        return Ok(());
    }

    let output_file = format!(
        "{}_time_plot.png",
        environment.settings.log_file.replace(".json", "")
    );
    println!("Generating generation time plot at: {}", output_file);

    // Create the plot
    let root = BitMapBackend::new(&output_file, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find min and max values for y-axis
    let min_y = 0.0; // Time can't be negative
    let max_y = environment
        .generation_time_history
        .iter()
        .map(|(_gen, time)| *time)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b))
        * 1.1; // Add 10% margin

    let max_gen = environment.generation_time_history.len() as u32;

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
            environment
                .generation_time_history
                .iter()
                .map(|(gen, time)| (*gen as u32, *time)),
            &BLUE,
        ))?
        .label("Generation Time (s)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Add a trend line (moving average) if we have enough data points
    if environment.generation_time_history.len() >= 3 {
        // Calculate moving average (window size of 3)
        let mut trend_data = Vec::new();
        for i in 1..environment.generation_time_history.len() - 1 {
            let avg_time = (environment.generation_time_history[i - 1].1
                + environment.generation_time_history[i].1
                + environment.generation_time_history[i + 1].1)
                / 3.0;
            trend_data.push((environment.generation_time_history[i].0 as u32, avg_time));
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

/// Calculate Euclidean distance between two neural networks
pub fn calculate_network_distance(nn1: &NeuralNetwork, nn2: &NeuralNetwork) -> f64 {
    // Check if networks have compatible architecture
    if nn1.layer_sizes != nn2.layer_sizes {
        return f64::MAX; // Return maximum distance if architectures differ
    }

    // Calculate distance between weights
    let mut total_squared_diff = 0.0;
    let mut total_elements = 0;

    // Compare weights matrices
    for (w1, w2) in nn1.weights.iter().zip(nn2.weights.iter()) {
        if w1.rows != w2.rows || w1.columns != w2.columns {
            return f64::MAX; // Return maximum distance if dimensions differ
        }

        for (v1, v2) in w1.values.iter().zip(w2.values.iter()) {
            total_squared_diff += (v1 - v2).powi(2);
            total_elements += 1;
        }
    }

    // Compare biases matrices
    for (b1, b2) in nn1.biases.iter().zip(nn2.biases.iter()) {
        if b1.rows != b2.rows || b1.columns != b2.columns {
            return f64::MAX; // Return maximum distance if dimensions differ
        }

        for (v1, v2) in b1.values.iter().zip(b2.values.iter()) {
            total_squared_diff += (v1 - v2).powi(2);
            total_elements += 1;
        }
    }

    // Return the root mean squared difference (normalized by number of weights)
    if total_elements > 0 {
        (total_squared_diff / total_elements as f64).sqrt() * total_elements as f64
    } else {
        0.0
    }
}

/// Plot population diversity metrics throughout evolution
pub fn plot_diversity_history(environment: &TrainingEnvironment) -> Result<()> {
    if environment.diversity_history.is_empty() {
        println!("No diversity data to plot");
        return Ok(());
    }

    let output_file = format!(
        "{}_diversity_plot.png",
        environment.settings.log_file.replace(".json", "")
    );
    println!("Generating diversity plot at: {}", output_file);

    // Create the plot
    let root = BitMapBackend::new(&output_file, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find min and max values for y-axis
    let min_y = 0.0; // Diversity can't be negative
    let max_y = environment
        .diversity_history
        .iter()
        .map(|(_gen, diversity)| *diversity)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b))
        * 1.1; // Add 10% margin

    let max_gen = environment.diversity_history.len() as u32;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Genetic Diversity over Generations",
            ("sans-serif", 30).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0u32..max_gen, min_y..max_y)?;

    chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Diversity (avg pairwise distance)")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    // Plot population diversity
    chart
        .draw_series(LineSeries::new(
            environment
                .diversity_history
                .iter()
                .map(|(gen, div)| (*gen as u32, *div)),
            &PURPLE,
        ))?
        .label("Population Diversity")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &PURPLE));

    // Add a trend line (moving average) if we have enough data points
    if environment.diversity_history.len() >= 3 {
        // Calculate moving average (window size of 3)
        let mut trend_data = Vec::new();
        for i in 1..environment.diversity_history.len() - 1 {
            let avg_diversity = (environment.diversity_history[i - 1].1
                + environment.diversity_history[i].1
                + environment.diversity_history[i + 1].1)
                / 3.0;
            trend_data.push((environment.diversity_history[i].0 as u32, avg_diversity));
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
    println!("Diversity plot generated at: {}", output_file);

    Ok(())
}

/// Plot benchmark performance over generations
pub fn plot_benchmark_history(environment: &TrainingEnvironment) -> Result<()> {
    if environment.benchmark_history.is_empty() {
        println!("No benchmark data to plot");
        return Ok(());
    }

    let output_file = format!(
        "{}_benchmark_plot.png",
        environment.settings.log_file.replace(".json", "")
    );
    println!("Generating benchmark performance plot at: {}", output_file);

    // Create the plot
    let root = BitMapBackend::new(&output_file, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find y-axis range (0.0 to 1.0 is reasonable for win rates)
    let min_y = 0.0;
    let max_y = 1.0;

    let max_gen = environment.benchmark_history.len() as u32;

    let mut chart = ChartBuilder::on(&root)
        .caption("Benchmark Performance", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0u32..max_gen, min_y..max_y)?;

    chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Win Rate")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    // Plot scores against random agent
    chart
        .draw_series(LineSeries::new(
            environment
                .benchmark_history
                .iter()
                .map(|(gen, random, _)| (*gen as u32, *random)),
            &BLUE,
        ))?
        .label("vs Random Agent")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Plot scores against simple agent
    chart
        .draw_series(LineSeries::new(
            environment
                .benchmark_history
                .iter()
                .map(|(gen, _, simple)| (*gen as u32, *simple)),
            &ORANGE,
        ))?
        .label("vs Forward Agent")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &ORANGE));

    // Add a reference line at 0.5 (break-even)
    chart
        .draw_series(LineSeries::new(
            vec![(0, 0.5), (max_gen, 0.5)],
            &BLACK.mix(0.3),
        ))?
        .label("Break-even")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK.mix(0.3)));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Benchmark plot generated at: {}", output_file);

    Ok(())
}

/// Plot how long each generation's champion survives
pub fn plot_champion_survival(environment: &TrainingEnvironment) -> Result<()> {
    if environment.champion_survival_history.is_empty() {
        println!("No champion survival data to plot");
        return Ok(());
    }

    let output_file = format!(
        "{}_champion_survival_plot.png",
        environment.settings.log_file.replace(".json", "")
    );
    println!("Generating champion survival plot at: {}", output_file);

    // Create the plot
    let root = BitMapBackend::new(&output_file, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    // Determine the maximum values for axes
    let max_gen = environment
        .champion_survival_history
        .iter()
        .map(|(gen, _)| *gen)
        .max()
        .unwrap_or(0) as u32
        + 5; // Add some margin

    let max_survival = environment
        .champion_survival_history
        .iter()
        .map(|(_, survival)| *survival)
        .max()
        .unwrap_or(0) as u32
        + 2; // Add some margin

    // Create chart
    let mut chart = ChartBuilder::on(&root)
        .caption("Champion Survival Duration", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0u32..max_gen, 0u32..max_survival)?;

    chart
        .configure_mesh()
        .x_desc("Generation Champion First Appeared")
        .y_desc("Generations Survived")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    // Plot bars showing survival duration
    chart.draw_series(
        environment
            .champion_survival_history
            .iter()
            .map(|(gen, survival)| {
                Rectangle::new(
                    [(*gen as u32, 0), (*gen as u32 + 1, *survival as u32)],
                    BLUE.filled(),
                )
            }),
    )?;

    // Add a line showing the average survival duration
    let avg_survival = environment
        .champion_survival_history
        .iter()
        .map(|(_, survival)| *survival as f64)
        .sum::<f64>()
        / environment.champion_survival_history.len() as f64;

    chart
        .draw_series(LineSeries::new(
            vec![(0, avg_survival as u32), (max_gen, avg_survival as u32)],
            RED.mix(0.7).stroke_width(2),
        ))?
        .label(format!("Average Survival: {:.1} generations", avg_survival))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED.mix(0.7)));

    // Instead, create points at the bottom of the existing chart
    let introduction_points: Vec<(u32, u32)> = environment
        .champion_survival_history
        .iter()
        .map(|(gen, _)| (*gen as u32, 0)) // Place them at y=0
        .collect();

    if !introduction_points.is_empty() {
        // Draw circles at the bottom to indicate champion introduction points
        chart
            .draw_series(
                introduction_points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 5, GREEN.mix(0.7).filled())),
            )?
            .label("New Champion Introduction")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN.mix(0.7)));

        // Calculate the rate of change (new champions per 10 generations)
        let generations_span = environment
            .champion_survival_history
            .iter()
            .map(|(gen, _)| *gen)
            .max()
            .unwrap_or(0) as f64
            - environment
                .champion_survival_history
                .iter()
                .map(|(gen, _)| *gen)
                .min()
                .unwrap_or(0) as f64;

        if generations_span > 0.0 {
            let change_rate =
                (environment.champion_survival_history.len() as f64 * 10.0) / generations_span;
            chart
                .draw_series(LineSeries::new(vec![], RED))?
                .label(format!(
                    "Champion turnover rate: {:.2} per 10 generations",
                    change_rate
                ));
        }
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    root.present()?;
    println!("Champion survival plot generated at: {}", output_file);

    Ok(())
}

/// Plot the percentage of games ending with a win in each generation
pub fn plot_win_statistics(environment: &TrainingEnvironment) -> Result<()> {
    if environment.win_statistics_history.is_empty() {
        println!("No win statistics data to plot");
        return Ok(());
    }

    let output_file = format!(
        "{}_win_statistics_plot.png",
        environment.settings.log_file.replace(".json", "")
    );
    println!("Generating win statistics plot at: {}", output_file);

    // Create the plot
    let root = BitMapBackend::new(&output_file, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    // Calculate win percentages
    let win_percentages: Vec<(u32, f64)> = environment
        .win_statistics_history
        .iter()
        .map(|(gen, wins, total)| {
            (
                *gen as u32,
                if *total > 0 {
                    100.0 * *wins as f64 / *total as f64
                } else {
                    0.0
                },
            )
        })
        .collect();

    // Find min and max values for y-axis
    let min_y = 0.0;
    let max_y = 100.0;

    let max_gen = environment.win_statistics_history.len() as u32;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Game Outcomes per Generation",
            ("sans-serif", 30).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0u32..max_gen, min_y..max_y)?;

    chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Win Percentage (%)")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    // Plot win percentage
    chart
        .draw_series(LineSeries::new(win_percentages, &GREEN.mix(0.8)))?
        .label("Games Ending in Win (%)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN.mix(0.8)));

    // Add bar chart showing actual win counts
    chart.draw_series(
        environment
            .win_statistics_history
            .iter()
            .map(|(gen, wins, total)| {
                let win_pct = if *total > 0 {
                    100.0 * *wins as f64 / *total as f64
                } else {
                    0.0
                };
                Rectangle::new(
                    [(*gen as u32, 0.0), ((*gen as f64 + 0.8) as u32, win_pct)],
                    GREEN.mix(0.3).filled(),
                )
            }),
    )?;

    // Add a reference line at 50%
    chart
        .draw_series(LineSeries::new(
            vec![(0, 50.0), (max_gen, 50.0)],
            &BLACK.mix(0.3),
        ))?
        .label("50% Reference")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK.mix(0.3)));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Win statistics plot generated at: {}", output_file);

    Ok(())
}
