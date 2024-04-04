import enum

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, Normalize


class ColorMap(enum.Enum):
    RedYellowGreen = enum.auto()
    RedGreenCustom = enum.auto()
    Unknown = enum.auto()


def red_yellow_green_cm():
    """Standard Matlab Red-Yellow-Green colormap.

    Returns:
        RYG colormap
    """
    return mpl.cm.RdYlGn


def custom_red_green_cm():
    """Customize colormap to return only shades of red and green.

    Returns:
        Custom colormap, Green and Red only
    """
    # Define color points for red and green
    colors = {"red":   [(0.0, 1.0, 1.0),
                        (1.0, 0.0, 0.0)],
              "green": [(0.0, 0.0, 0.0),
                        (1.0, 1.0, 1.0)],
              "blue":  [(0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)]}
    # Create colormap
    return LinearSegmentedColormap("custom_rg", colors)


class ColorMapGen:
    def __init__(self, name: ColorMap = ColorMap.Unknown):
        self.norm = mpl.colors.Normalize(vmin=0, vmax=1)
        if name == ColorMap.RedYellowGreen:
            self.cmap = red_yellow_green_cm()
        elif name == ColorMap.RedGreenCustom:
            self.cmap = custom_red_green_cm()
        else:
            raise ValueError(f"Unsupported colormap name: {name}!")

    def generate_color(self, probability: float):
        assert 0. <= probability <= 1.0, f"Probability value must be in range [0., 1.], got {probability=}"
        color = self.cmap(self.norm(probability))
        return mpl.colors.to_hex(color)
