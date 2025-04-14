use super::constants::*;
use bevy::prelude::*;
use quoridor::vector::Vector;

#[derive(Component)]
struct Tile;

pub fn spawn_tile(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    tile_pos: Vector,
) {
    let screen_pos: Vec2 = calculate_screen_pos(tile_pos);
    commands.spawn((
        Tile,
        Mesh2d(meshes.add(Rectangle::new(TILE_WIDTH, TILE_WIDTH))),
        MeshMaterial2d(materials.add(Color::from(TILE_COLOR))),
        Transform::from_xyz(screen_pos.x, screen_pos.y, TILE_Z),
    ));
}

pub fn calculate_screen_pos(tile_pos: Vector) -> Vec2 {
    let screen_x: f32 = (tile_pos.x - HALF_TILE_GRID_SIZE) as f32 * TOTAL_WIDTH;
    let screen_y: f32 = (HALF_TILE_GRID_SIZE - tile_pos.y) as f32 * TOTAL_WIDTH;
    Vec2::new(screen_x, screen_y)
}
