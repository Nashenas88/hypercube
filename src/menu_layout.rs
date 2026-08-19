//! Menu content shared into `menu_overlay.rs`'s `MenuBar` widget, so the
//! label/action pairing for each top-level menu is defined in one place
//! away from the low-level widget code.

use crate::app::{AABBMode, Message, RenderMode, reveal_button_label};

/// Fixed per-button width, so a dropdown's position can be computed from
/// its button's index in `TopMenu::ALL` without measuring rendered text.
pub(crate) const BUTTON_WIDTH: f32 = 70.0;
pub(crate) const BUTTON_SPACING: f32 = 5.0;
pub(crate) const BAR_HEIGHT: f32 = 32.0;
pub(crate) const BAR_PADDING: f32 = 5.0;

/// Move count for the "Scramble" menu item. 25 mixes a 27-piece side
/// several times over (180-degree edge and 120-degree corner turns disturb
/// most of a side per move), enough that the puzzle reads as thoroughly
/// shuffled without an excessive click-to-solved feel for manual play.
const SCRAMBLE_MOVE_COUNT: u32 = 25;

/// A top-level entry in the menu bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TopMenu {
    File,
    Puzzle,
    View,
    Help,
}

impl TopMenu {
    pub(crate) const ALL: [TopMenu; 4] =
        [TopMenu::File, TopMenu::Puzzle, TopMenu::View, TopMenu::Help];

    pub(crate) fn label(self) -> &'static str {
        match self {
            TopMenu::File => "File",
            TopMenu::Puzzle => "Puzzle",
            TopMenu::View => "View",
            TopMenu::Help => "Help",
        }
    }
}

/// One clickable row in a dropdown.
#[derive(Debug, Clone)]
pub(crate) struct MenuItem {
    pub(crate) label: String,
    pub(crate) message: Message,
    /// Whether this row reflects the app's current state (Render Mode/AABB
    /// Mode entries) - plain actions are never marked.
    pub(crate) selected: bool,
}

impl std::fmt::Display for MenuItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.selected && !matches!(self.message, Message::NoOp) {
            write!(f, "\u{2713} {}", self.label)
        } else {
            write!(f, "   {}", self.label)
        }
    }
}

pub(crate) fn file_items() -> Vec<MenuItem> {
    vec![
        MenuItem {
            label: "Save".to_string(),
            message: Message::SavePuzzle,
            selected: false,
        },
        MenuItem {
            label: "Load".to_string(),
            message: Message::LoadPuzzle,
            selected: false,
        },
        MenuItem {
            label: "Quit".to_string(),
            message: Message::Quit,
            selected: false,
        },
    ]
}

/// A blank row used to visually separate groups of Puzzle menu items.
fn spacer_item() -> MenuItem {
    MenuItem {
        label: String::new(),
        message: Message::NoOp,
        selected: false,
    }
}

pub(crate) fn puzzle_items(revealed: bool, reveal_animating: bool) -> Vec<MenuItem> {
    vec![
        MenuItem {
            label: "Reset".to_string(),
            message: Message::Reset,
            selected: false,
        },
        spacer_item(),
        MenuItem {
            label: "1 Random Move".to_string(),
            message: Message::RandomMoves(1),
            selected: false,
        },
        MenuItem {
            label: "2 Random Moves".to_string(),
            message: Message::RandomMoves(2),
            selected: false,
        },
        MenuItem {
            label: "3 Random Moves".to_string(),
            message: Message::RandomMoves(3),
            selected: false,
        },
        MenuItem {
            label: "Scramble".to_string(),
            message: Message::RandomMoves(SCRAMBLE_MOVE_COUNT),
            selected: false,
        },
        spacer_item(),
        MenuItem {
            label: reveal_button_label(revealed, reveal_animating).to_string(),
            message: Message::ToggleReveal,
            selected: false,
        },
    ]
}

pub(crate) fn view_items(render_mode: RenderMode, aabb_mode: AABBMode) -> Vec<MenuItem> {
    RenderMode::ALL
        .into_iter()
        .map(|mode| MenuItem {
            label: format!("Render Mode: {mode}"),
            selected: mode == render_mode,
            message: Message::RenderMode(mode),
        })
        .chain(AABBMode::ALL.into_iter().map(|mode| MenuItem {
            label: format!("{mode}"),
            selected: mode == aabb_mode,
            message: Message::AABBMode(mode),
        }))
        .collect()
}

pub(crate) fn help_items() -> Vec<MenuItem> {
    vec![MenuItem {
        label: "About".to_string(),
        message: Message::OpenAbout,
        selected: false,
    }]
}
