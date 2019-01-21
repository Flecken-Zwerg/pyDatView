import unittest
from pydatview.common import *



class TestCommon(unittest.TestCase):
    def assertEqual(self, first, second, msg=None):
        #print('>',first,'<',' >',second,'<')
        super(TestCommon, self).assertEqual(first, second, msg)
    
    def test_unit(self):
        self.assertEqual(unit   ('speed [m/s]'),'m/s'  )
        self.assertEqual(unit   ('speed [m/s' ),'m/s'  ) # ...
        self.assertEqual(no_unit('speed [m/s]'),'speed')
    
    def test_ellude(self):
        self.assertListEqual(ellude_common(['AAA','ABA']),['A','B'])

        # unit test for #25
        S=ellude_common(['A.txt','A_.txt'])
        if any([len(s)<=1 for s in S]):
            raise Exception('[FAIL] ellude common with underscore difference, Bug #25')

 
if __name__ == '__main__':
    unittest.main()
