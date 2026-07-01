
class PlotItem:
    def __init__(self,
                 name: str = None,
                 source_name: str = None,
                 color: str = None,
                 linestyle: str = None,
                 **kwargs) -> None:
        self.name = name
        self.source_name = source_name
        self.color = color
        self.linestyle = linestyle
        self.prefix_name = kwargs.get("prefix_name", None)
