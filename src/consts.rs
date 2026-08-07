use iced::mouse::Button;

#[cfg(false)]
pub(crate) const MOUSE_KEY: Button = Button::Right;
#[cfg(true)]
pub(crate) const MOUSE_KEY: Button = Button::Left;