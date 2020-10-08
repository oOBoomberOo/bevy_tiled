use crate::{
    map::{Chunk, Map},
    Layer, Tile, TilesetLayer,
};
use anyhow::Result;
use bevy::{
    asset::AssetLoader,
    prelude::Mesh,
    render::{mesh::VertexAttribute, pipeline::PrimitiveTopology},
};
use glam::{Vec2, Vec4};

use std::{io::BufReader, path::Path};

#[derive(Default)]
pub struct TiledMapLoader;

impl TiledMapLoader {
    fn remove_tile_flags(tile: u32) -> u32 {
        tile & !ALL_FLIP_FLAGS
    }
}

const FLIPPED_HORIZONTALLY_FLAG: u32 = 0x80000000;
const FLIPPED_VERTICALLY_FLAG: u32 = 0x40000000;
const FLIPPED_DIAGONALLY_FLAG: u32 = 0x20000000;
const ALL_FLIP_FLAGS: u32 =
    FLIPPED_HORIZONTALLY_FLAG | FLIPPED_VERTICALLY_FLAG | FLIPPED_DIAGONALLY_FLAG;

impl AssetLoader<Map> for TiledMapLoader {
    fn from_bytes(&self, asset_path: &Path, bytes: Vec<u8>) -> Result<Map> {
        let map = tiled::parse_with_path(BufReader::new(bytes.as_slice()), asset_path).unwrap();

        let target_chunk_x = 32;
        let target_chunk_y = 32;

        let chunk_size_x = (map.width as f32 / target_chunk_x as f32).ceil().max(1.0) as usize;
        let chunk_size_y = (map.height as f32 / target_chunk_y as f32).ceil().max(1.0) as usize;
        let tile_size = Vec2::new(map.tile_width as f32, map.tile_height as f32);

        let create_tileset_layer =
            |tileset: &tiled::Tileset, layer_tiles: &[Vec<tiled::LayerTile>]| {
                let tile_width = tileset.tile_width as f32;
                let tile_height = tileset.tile_height as f32;
                let image = tileset.images.first().unwrap();
                let texture_width = image.width as f32;
                let texture_height = image.height as f32;
                let columns = (texture_width / tile_width).floor();

                let chunks = double_vector(0..chunk_size_x, 0..chunk_size_y, |chunk_x, chunk_y| {
                    let tiles =
                        double_vector(0..target_chunk_x, 0..target_chunk_y, |tile_x, tile_y| {
                            let lookup_x = (chunk_x * target_chunk_x) + tile_x;
                            let lookup_y = (chunk_y * target_chunk_y) + tile_y;

                            // Get chunk tile.
                            let result = if lookup_x < map.width as usize
                                && lookup_y < map.height as usize
                            {
                                // New Tiled crate code:
                                let map_tile = &layer_tiles[lookup_y][lookup_x];

                                let tile = map_tile.gid;
                                if tile < tileset.first_gid
                                    || tile >= tileset.first_gid + tileset.tilecount.unwrap()
                                {
                                    return None;
                                }

                                let tile =
                                    (Self::remove_tile_flags(tile) - tileset.first_gid) as f32;

                                // This calculation is much simpler we only care about getting the remainder
                                // and multiplying that by the tile width.
                                let sprite_sheet_x: f32 = (tile % columns * tile_width).floor();

                                // Calculation here is (tile / columns).round_down * tile_height
                                // Example: tile 30 / 28 columns = 1.0714 rounded down to 1 * 16 tile_height = 16 Y
                                // which is the 2nd row in the sprite sheet.
                                // Example2: tile 10 / 28 columns = 0.3571 rounded down to 0 * 16 tile_height = 0 Y
                                // which is the 1st row in the sprite sheet.
                                let sprite_sheet_y: f32 = (tile / columns).floor() * tile_height;

                                // Calculate positions
                                let (start, end) = {
                                    let size = Vec2::new(tile_width, tile_height);
                                    let pos = Vec2::new(lookup_x as f32, lookup_y as f32);
                                    calculate_orientation(map.orientation, size, pos)
                                };

                                // Calculate UV:
                                let mut start_u: f32 = sprite_sheet_x / texture_width;
                                let mut end_u: f32 = (sprite_sheet_x + tile_width) / texture_width;
                                let mut start_v: f32 = sprite_sheet_y / texture_height;
                                let mut end_v: f32 =
                                    (sprite_sheet_y + tile_height) / texture_height;

                                if map_tile.flip_h {
                                    std::mem::swap(&mut start_u, &mut end_u);
                                }
                                if map_tile.flip_v {
                                    std::mem::swap(&mut start_v, &mut end_v);
                                }

                                Tile {
                                    tile_id: map_tile.gid,
                                    pos: Vec2::new(*tile_x as f32, *tile_y as f32),
                                    vertex: Vec4::new(start.x(), start.y(), end.x(), end.y()),
                                    uv: Vec4::new(start_u, start_v, end_u, end_v),
                                }
                            } else {
                                // Empty tile
                                Tile {
                                    tile_id: 0,
                                    pos: Vec2::new(*tile_x as f32, *tile_y as f32),
                                    vertex: Vec4::new(0.0, 0.0, 0.0, 0.0),
                                    uv: Vec4::new(0.0, 0.0, 0.0, 0.0),
                                }
                            };
                            Some(result)
                        });

                    let chunk = Chunk {
                        position: Vec2::new(*chunk_x as f32, *chunk_y as f32),
                        tiles,
                    };

                    Some(chunk)
                });

                TilesetLayer {
                    tile_size: Vec2::new(tile_width, tile_height),
                    chunks,
                    tileset_guid: tileset.first_gid,
                }
            };

        let iter = map
            .layers
            .iter()
            .filter(|layer| layer.visible)
            .map(get_tiles);

        let mut layers: Vec<Layer> = Vec::new();

        for tiles in iter {
            let tileset_layers = map
                .tilesets
                .iter()
                .map(|tileset| create_tileset_layer(tileset, tiles))
                .collect();

            let layer = Layer { tileset_layers };
            layers.push(layer);
        }

        let iter = layers
            .iter()
            .map(|layer| layer.tileset_layers.iter())
            .enumerate()
            .flat_map(|(n, tilesets)| tilesets.map(move |layer| (n, layer)));

        let mut meshes = Vec::new();

        for (layer_id, tileset_layer) in iter {
            for chunk in tileset_layer.chunks.iter().flatten() {
                let mut positions = Vec::new();
                let mut normals = Vec::new();
                let mut uvs = Vec::new();
                let mut indices = Vec::new();

                let mut i = 0;
                for tile in chunk.tiles.iter().flatten() {
                    if tile.tile_id < tileset_layer.tileset_guid {
                        continue;
                    }

                    // X, Y
                    positions.push([tile.vertex.x(), tile.vertex.y(), 0.0]);
                    normals.push([0.0, 0.0, 1.0]);
                    uvs.push([tile.uv.x(), tile.uv.w()]);

                    // X, Y + 1
                    positions.push([tile.vertex.x(), tile.vertex.w(), 0.0]);
                    normals.push([0.0, 0.0, 1.0]);
                    uvs.push([tile.uv.x(), tile.uv.y()]);

                    // X + 1, Y + 1
                    positions.push([tile.vertex.z(), tile.vertex.w(), 0.0]);
                    normals.push([0.0, 0.0, 1.0]);
                    uvs.push([tile.uv.z(), tile.uv.y()]);

                    // X + 1, Y
                    positions.push([tile.vertex.z(), tile.vertex.y(), 0.0]);
                    normals.push([0.0, 0.0, 1.0]);
                    uvs.push([tile.uv.z(), tile.uv.w()]);

                    let mut new_indices = vec![i + 0, i + 2, i + 1, i + 0, i + 3, i + 2];
                    indices.append(&mut new_indices);

                    i += 4;
                }

                if !positions.is_empty() {
                    let mesh = Mesh {
                        primitive_topology: PrimitiveTopology::TriangleList,
                        attributes: vec![
                            VertexAttribute::position(positions),
                            VertexAttribute::normal(normals),
                            VertexAttribute::uv(uvs),
                        ],
                        indices: Some(indices),
                    };
                    meshes.push((layer_id as u32, tileset_layer.tileset_guid, mesh));
                }
            }
        }

        let map = Map {
            map,
            meshes,
            layers,
            tile_size,
            image_folder: asset_path.parent().unwrap().to_str().unwrap().to_string(),
        };

        Ok(map)
    }

    fn extensions(&self) -> &[&str] {
        &["tmx"]
    }
}

fn get_tiles(layer: &tiled::Layer) -> &[Vec<tiled::LayerTile>] {
    match &layer.tiles {
        tiled::LayerData::Finite(tiles) => tiles.as_slice(),
        _ => panic!("Infinite maps are not supported"),
    }
}

fn calculate_orientation(orientation: tiled::Orientation, size: Vec2, pos: Vec2) -> (Vec2, Vec2) {
    let center = match orientation {
        tiled::Orientation::Orthogonal => Map::project_ortho(pos, size.x(), size.y()),
        tiled::Orientation::Isometric => Map::project_iso(pos, size.x(), size.y()),
        _ => panic!("Unsupported orientation {:?}", orientation),
    };

    let start = Vec2::new(center.x() - size.x() / 2.0, center.y() - size.y() / 2.0);
    let end = Vec2::new(center.x() + size.x() / 2.0, center.y() + size.y() / 2.0);

    (start, end)
}

/// A helper function for creating double vector from a given range
fn double_vector<A, B, F, T>(
    first: impl IntoIterator<Item = A>,
    second: impl IntoIterator<Item = B> + Clone,
    f: F,
) -> Vec<Vec<T>>
where
    F: Fn(&A, &B) -> Option<T>,
{
    let mut result = Vec::new();
    for x in first {
        let mut local = Vec::new();
        for y in second.clone() {
            if let Some(result) = f(&x, &y) {
                local.push(result);
            }
        }
        result.push(local);
    }
    result
}
