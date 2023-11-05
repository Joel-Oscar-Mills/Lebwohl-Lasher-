import numpy as np


def update_angles(angles,rangles,rows,offset):

    parity = offset%2
    old_angles = angles

    # Perturb the angles (for on-parity sites)
    if parity == 0:
        angles[(parity):rows+1:2,parity::2] += rangles[parity::2,parity::2]
        angles[(1-parity):rows+1:2,(1-parity)::2] += rangles[(1-parity)::2,(1-parity)::2]
    else:
        angles[(parity):rows+1:2,(1-parity)::2] += rangles[parity::2,(1-parity)::2]
        angles[(1-parity):rows+1:2,parity::2] += rangles[(1-parity)::2,parity::2]

    print(angles)

    return old_angles, angles


def main():

    NMAX = 50
    Ts = 0.5
    scale = 0.1+Ts
    rows = NMAX
    offset = 1
    angles = np.full((NMAX,NMAX),2)
    rangles = np.full((NMAX,NMAX),1)

    old_angles, angles = update_angles(angles,rangles,rows,offset)


    

if __name__ == "__main__":
    main() 