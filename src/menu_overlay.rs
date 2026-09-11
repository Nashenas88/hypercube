//! The File/Puzzle/View/Help menu bar, built on `overlay::menu::Menu` (the
//! same lower-level primitive `PickList`/`ComboBox` use internally) via a
//! hand-written `advanced::Widget` impl (`MenuBar`) modeled closely on
//! `iced_widget::pick_list::PickList`'s own `Widget` impl.
//!
//! No backdrop/`mouse_area` is needed to close on an outside click: exactly
//! like `PickList`, `MenuBar::update` unconditionally closes on any left
//! click it receives while a menu is open (the dropdown's own
//! `overlay::menu::Menu` consumes clicks that land on it first, so this only
//! fires for clicks elsewhere) - that behavior comes from the widget tree's
//! own event routing for free.

use std::cell::Cell;

use iced::advanced::widget::{Tree, tree};
use iced::advanced::{self, Clipboard, Layout, Shell, layout, mouse, overlay, renderer, text};
use iced::overlay::menu::{self, Menu};
use iced::{Element, Event, Length, Point, Rectangle, Size, Vector};

use crate::app::{AABBMode, Message, RenderMode};
use crate::menu_layout::{
    self, BAR_HEIGHT, BAR_PADDING, BUTTON_SPACING, BUTTON_WIDTH, MenuItem, TopMenu,
};

/// The File/Puzzle/View/Help menu bar, built from `render_mode`/`aabb_mode`
/// (for the View menu's checkmarks), `revealed`/`reveal_animating` (for the
/// Puzzle menu's Reveal/Hide label), `solving` (for its Solve/Stop Solving
/// item), and `debug_mode` (which gates whether View is present) -
/// snapshotted fresh each `view()` call, same as `HypercubeShaderProgram`.
pub(crate) fn bar<'a>(
    debug_mode: bool,
    render_mode: RenderMode,
    aabb_mode: AABBMode,
    revealed: bool,
    reveal_animating: bool,
    solving: bool,
) -> Element<'a, Message> {
    let visible: Vec<TopMenu> = TopMenu::ALL
        .into_iter()
        .filter(|top| *top != TopMenu::View || debug_mode)
        .collect();

    Element::new(MenuBar {
        file_items: menu_layout::file_items(),
        puzzle_items: menu_layout::puzzle_items(revealed, reveal_animating, solving),
        view_items: menu_layout::view_items(render_mode, aabb_mode),
        help_items: menu_layout::help_items(),
        visible,
        menu_class: <iced::Theme as menu::Catalog>::default(),
    })
}

struct MenuBar<'a> {
    visible: Vec<TopMenu>,
    file_items: Vec<MenuItem>,
    puzzle_items: Vec<MenuItem>,
    view_items: Vec<MenuItem>,
    help_items: Vec<MenuItem>,
    menu_class: <iced::Theme as menu::Catalog>::Class<'a>,
}

/// Per-top-level-menu widget-tree state: its own `overlay::menu::Menu`
/// state plus which row (if any) is hovered.
#[derive(Default)]
struct MenuState {
    menu: menu::State,
    hovered_option: Option<usize>,
}

#[derive(Default)]
struct State {
    file: MenuState,
    puzzle: MenuState,
    view: MenuState,
    help: MenuState,
    /// Which top-level menu is open, if any.
    open: Cell<Option<TopMenu>>,
}

impl<'a> MenuBar<'a> {
    fn items(&self, top: TopMenu) -> &Vec<MenuItem> {
        match top {
            TopMenu::File => &self.file_items,
            TopMenu::Puzzle => &self.puzzle_items,
            TopMenu::View => &self.view_items,
            TopMenu::Help => &self.help_items,
        }
    }

    fn button_at(&self, layout: Layout<'_>, cursor: mouse::Cursor) -> Option<TopMenu> {
        self.visible
            .iter()
            .zip(layout.children())
            .find(|(_, child)| cursor.is_over(child.bounds()))
            .map(|(top, _)| *top)
    }
}

impl<'a> advanced::Widget<Message, iced::Theme, iced::Renderer> for MenuBar<'a> {
    fn size(&self) -> Size<Length> {
        Size {
            width: Length::Shrink,
            height: Length::Fixed(BAR_HEIGHT),
        }
    }

    fn layout(
        &mut self,
        _tree: &mut Tree,
        _renderer: &iced::Renderer,
        _limits: &layout::Limits,
    ) -> layout::Node {
        let mut children = Vec::with_capacity(self.visible.len());
        let mut x = BAR_PADDING;
        for _ in &self.visible {
            children.push(
                layout::Node::new(Size::new(BUTTON_WIDTH, BAR_HEIGHT)).move_to(Point::new(x, 0.0)),
            );
            x += BUTTON_WIDTH + BUTTON_SPACING;
        }

        let width = (x - BUTTON_SPACING + BAR_PADDING).max(0.0);
        layout::Node::with_children(Size::new(width, BAR_HEIGHT), children)
    }

    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(State::default())
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut iced::Renderer,
        _theme: &iced::Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        use advanced::Renderer as _;
        use advanced::text::Renderer as _;

        let state = tree.state.downcast_ref::<State>();

        for (top, child) in self.visible.iter().zip(layout.children()) {
            let bounds = child.bounds();
            let is_open = state.open.get() == Some(*top);
            let is_hovered = cursor.is_over(bounds);

            let background = if is_open {
                iced::Color::from_rgb(0.32, 0.32, 0.38)
            } else if is_hovered {
                iced::Color::from_rgb(0.24, 0.24, 0.28)
            } else {
                iced::Color::TRANSPARENT
            };

            renderer.fill_quad(
                renderer::Quad {
                    bounds,
                    ..renderer::Quad::default()
                },
                background,
            );

            renderer.fill_text(
                text::Text {
                    content: top.label().to_string(),
                    bounds: bounds.size(),
                    size: renderer.default_size(),
                    line_height: text::LineHeight::default(),
                    font: renderer.default_font(),
                    align_x: text::Alignment::Center,
                    align_y: iced::alignment::Vertical::Center,
                    shaping: text::Shaping::Basic,
                    wrapping: text::Wrapping::default(),
                },
                Point::new(bounds.center_x(), bounds.center_y()),
                iced::Color::WHITE,
                bounds,
            );
        }
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _renderer: &iced::Renderer,
        _clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        _viewport: &Rectangle,
    ) {
        let state = tree.state.downcast_mut::<State>();

        if let Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) = event {
            if state.open.get().is_some() {
                state.open.set(None);
                shell.request_redraw();
                shell.capture_event();
            } else if let Some(top) = self.button_at(layout, cursor) {
                state.open.set(Some(top));
                shell.request_redraw();
                shell.capture_event();
            }
        }
    }

    fn overlay<'b>(
        &'b mut self,
        tree: &'b mut Tree,
        layout: Layout<'_>,
        _renderer: &iced::Renderer,
        viewport: &Rectangle,
        translation: Vector,
    ) -> Option<overlay::Element<'b, Message, iced::Theme, iced::Renderer>> {
        let state = tree.state.downcast_mut::<State>();
        let open = state.open.get()?;
        let index = self.visible.iter().position(|top| *top == open)?;
        let bounds = layout.child(index).bounds();
        let items = self.items(open);

        // Taken before the match below borrows `state.file`/`.puzzle`/`.view`/`.help` mutably.
        let open_cell = &state.open;

        let (menu_state, hovered) = match open {
            TopMenu::File => (&mut state.file.menu, &mut state.file.hovered_option),
            TopMenu::Puzzle => (&mut state.puzzle.menu, &mut state.puzzle.hovered_option),
            TopMenu::View => (&mut state.view.menu, &mut state.view.hovered_option),
            TopMenu::Help => (&mut state.help.menu, &mut state.help.hovered_option),
        };

        let menu = Menu::new(
            menu_state,
            items,
            hovered,
            move |item: MenuItem| {
                // Ignore spacers
                if !matches!(item.message, Message::NoOp) {
                    open_cell.set(None);
                }
                item.message
            },
            None,
            &self.menu_class,
        )
        .width(220.0);

        Some(menu.overlay(
            bounds.position() + translation,
            *viewport,
            bounds.height,
            Length::Shrink,
        ))
    }
}
