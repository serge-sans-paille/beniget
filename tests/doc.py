import doctest
import unittest
import os

import beniget


class TestDoctest(unittest.TestCase):
    def test_beniget_documentation(self):
        failed, _ = doctest.testmod(beniget.beniget)
        self.assertEqual(failed, 0)

    def test_beniget_readme(self):
        failed, _ = doctest.testfile(os.path.join("..", "README.rst"))
        self.assertEqual(failed, 0)
