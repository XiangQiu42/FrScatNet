from .scattering1d.frontend.torch_frontend import ScatteringTorch1D as Scattering1D
from .scattering2d.frontend.torch_frontend import ScatteringTorch2D as Scattering2D
from .scattering3d.frontend.torch_frontend \
        import HarmonicScatteringTorch3D as HarmonicScattering3D
# add our new fractional scattering2D (with pytorch)
from .scattering2d.frontend.torch_fr_frontend import FrScatteringTorch2D as Fr_Scattering2D

Scattering1D.__module__ = 'kymatio.torch'
Scattering1D.__name__ = 'Scattering1D'

Scattering2D.__module__ = 'kymatio.torch'
Scattering2D.__name__ = 'Scattering2D'

HarmonicScattering3D.__module__ = 'kymatio.torch'
HarmonicScattering3D.__name__ = 'HarmonicScattering3D'

Fr_Scattering2D.__module__ = 'kymatio.torch'
Fr_Scattering2D.__name__ = 'FrScattering2D'

__all__ = ['Scattering1D', 'Scattering2D', 'HarmonicScattering3D', 'Fr_Scattering2D']
