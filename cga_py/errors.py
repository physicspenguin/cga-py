class GeometryError(Exception):

    """Error class for purely geometrically caused errors"""

    def __init__(self, *args):
        if args:
            self.error_message = args[0]
        else:
            self.error_message = None

    def __str__(self):  # pragma: no cover

        if self.error_message:
            return "GeometryError , {0} ".format(self.error_message)
        else:
            return "GeometryError has been raised"
