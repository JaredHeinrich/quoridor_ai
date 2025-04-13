use super::constants::*;
use bevy::prelude::*;
use quoridor::vector::Vector;

#[derive(Component)]
struct Pawn;

pub fn spawn_pawn(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    pawn_pos: Vector,
    color: Srgba,
) {
    let screen_pos: Vec2 = super::tile::calculate_screen_pos(pawn_pos);
    commands.spawn((
        Pawn,
        Mesh2d(meshes.add(Circle::new(PAWN_RADIUS))),
        MeshMaterial2d(materials.add(Color::from(color))),
        Transform::from_xyz(screen_pos.x, screen_pos.y, PAWN_Z),
    ));
}
