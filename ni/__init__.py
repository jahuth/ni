"""NI toolbox
.. module:: ni
   :platform: Unix
   :synopsis: Neuroinformatics Toolbox

.. moduleauthor:: Jacob Huth

"""

import tools
import model
import data

from tools.html_view import View
from tools.statcollector import StatCollector
from data.data import Data, merge
from ni.tools.project import figure
from ni.tools.pickler import load
