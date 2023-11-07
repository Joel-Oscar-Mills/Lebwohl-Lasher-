import sys
from LebwohlLasher_cython import main

if len(sys.argv) == 1:
    main(sys.argv[1])

else:
    print("Usage: python {}".format(sys.argv[0]))