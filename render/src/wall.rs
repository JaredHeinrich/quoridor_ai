use bevy::prelude::*;
use quoridor::vector::Vector;
use quoridor::wall::Orientation;

use super::constants::*;

#[derive(Component)]
pub struct Wall;

pub fn spawn_wall(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    wall_pos: Vector,
    wall_orientation: Orientation,
) {
    let size: Vec2 = match wall_orientation {
        Orientation::Vertical => Vec2::new(WALL_WIDTH, WALL_LENGTH),
        Orientation::Horizontal => Vec2::new(WALL_LENGTH, WALL_WIDTH),
    };
    let screen_pos: Vec2 = calculate_screen_pos(wall_pos);
    commands.spawn((
        Wall,
        Mesh2d(meshes.add(Rectangle::new(size.x, size.y))),
        MeshMaterial2d(materials.add(Color::from(WALL_COLOR))),
        Transform::from_xyz(screen_pos.x, screen_pos.y, WALL_Z),
    ));
}

fn calculate_screen_pos(wall_pos: Vector) -> Vec2 {
    let screen_x: f32 = (wall_pos.x - HALF_TILE_GRID_SIZE) as f32 * TOTAL_WIDTH + OFFSET;
    let screen_y: f32 = (HALF_TILE_GRID_SIZE - wall_pos.y) as f32 * TOTAL_WIDTH - OFFSET;
    Vec2::new(screen_x, screen_y)
}

pub fn clear_walls(
    commands: &mut Commands,
    walls: &Query<Entity, With<Wall>>,
) {
    for wall in walls.iter() {
        commands.entity(wall).despawn();
    }
}
