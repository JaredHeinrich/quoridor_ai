use bevy::{
    color::palettes::css::{BROWN, GRAY, WHITE, YELLOW},
    prelude::*,
};
pub const TILE_GRID_SIZE: i16 = 9;
pub const HALF_TILE_GRID_SIZE: i16 = TILE_GRID_SIZE / 2;
pub const WALL_GRID_SIZE: i16 = TILE_GRID_SIZE - 1;
pub const WALL_WIDTH: f32 = 15.;
pub const HALF_WALL_WIDTH: f32 = WALL_WIDTH / 2.;
pub const WALL_LENGTH: f32 = 2. * TILE_WIDTH + WALL_WIDTH;
pub const WALL_Z: f32 = 1.;
pub const TILE_WIDTH: f32 = 45.;
pub const HALF_TILE_WIDTH: f32 = TILE_WIDTH / 2.;
pub const TILE_Z: f32 = 1.;
pub const OFFSET: f32 = HALF_TILE_WIDTH + HALF_WALL_WIDTH;
pub const TOTAL_WIDTH: f32 = WALL_WIDTH + TILE_WIDTH;
pub const HALF_TOTAL_WIDTH: f32 = TOTAL_WIDTH / 2.;
pub const TOTAL_BOARD_WIDTH: f32 =
    TILE_GRID_SIZE as f32 * TILE_WIDTH + (TILE_GRID_SIZE - 1) as f32 * WALL_WIDTH;
pub const HALF_BOARD_WIDTH: f32 = TOTAL_BOARD_WIDTH / 2.;
pub const BOARD_Z: f32 = 0.;
pub const PAWN_DIAMETER: f32 = 40.;
pub const PAWN_RADIUS: f32 = PAWN_DIAMETER / 2.;
pub const PAWN_Z: f32 = 2.;
pub const WALL_COLOR: Srgba = BROWN;
pub const WALL_PREVIEW_COLOR: Srgba = YELLOW;
pub const TILE_COLOR: Srgba = WHITE;
pub const BOARD_COLOR: Srgba = GRAY;
