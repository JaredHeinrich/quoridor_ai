use super::constants::*;
use super::tile::spawn_tile;
use bevy::prelude::*;
use quoridor::vector::Vector;

pub fn spawn_board(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
) {
    spawn_board_background(commands, meshes, materials);
    for x in 0..TILE_GRID_SIZE {
        for y in 0..TILE_GRID_SIZE {
            spawn_tile(commands, meshes, materials, Vector::new(x, y));
        }
    }
}

fn spawn_board_background(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(TOTAL_BOARD_WIDTH, TOTAL_BOARD_WIDTH))),
        MeshMaterial2d(materials.add(Color::from(BOARD_COLOR))),
        Transform::from_xyz(0., 0., BOARD_Z),
    ));
}
