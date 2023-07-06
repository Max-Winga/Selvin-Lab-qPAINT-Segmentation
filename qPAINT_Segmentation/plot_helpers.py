
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

def plot_scale_bar(nm_per_pixel):
    """Adds a scale bar to the current plot

    Args:
        nm_per_pixel (int or float): The conversion ratio for scaling
    """
    fontprops = fm.FontProperties(size=8)
    scalebar = AnchoredSizeBar(plt.gca().transData,
                            1000/nm_per_pixel,  # length of scale bar
                            '1 micron',  # label
                            'lower right',  # position
                            pad=0.1,
                            color='white',
                            frameon=False,
                            size_vertical=0.05,
                            fontproperties=fontprops)
    plt.gca().add_artist(scalebar)

class PlotColors():
    """
    A class used to cycle through a set of colors for plotting.

    Attributes:
        colors (list): The list of colors to cycle through.
        current_index (int): The current index in the colors list.

    Methods:
        get_next_color(): Returns the next color in the list, looping back to the start if 
                          the end is reached.
        __getitem__(idx): Returns the color at the specified index.
    """
    def __init__(self, colors):
        """Initializes the PlotColors class.

        Args:
            colors (list): The list of colors to cycle through.
        """
        self.colors = colors
        self.current_index = 0
    
    def get_next_color(self):
        """Returns the next color in the list, looping back to the start if the end is reached.

        Returns:
            str: The next color in the list.
        """
        color_to_return = self.colors[self.current_index]
        if self.current_index == len(self.colors) - 1:
            self.current_index = 0
        else:
            self.current_index += 1
        return color_to_return
    
    def __getitem__(self, idx):
        """Returns the color at the specified index.

        Args:
            idx (int): The index of the color to access.

        Returns:
            str: The color at the specified index.
        """
        return self.colors[idx]