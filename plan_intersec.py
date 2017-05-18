import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
following 
http://stackoverflow.com/questions/16025620/finding-the-line-along-the-the-intersection-of-two-planes
'''


########################################################
def get_plan_coeff(normal,point):
    plan_coeffs = np.zeros(4)
    plan_coeffs[:3] = np.array(normal)/np.sqrt((normal**2).sum())
    plan_coeffs[-1] = -1 *  np.dot(plan_coeffs[:3],point)
    return plan_coeffs
    


########################################################
def intersectPlanes(p1, p2):
    '''
    Algorithm taken from http://geomalgorithms.com/a05-_intersect-1.html. See the
    section 'Intersection of 2 Planes' and specifically the subsection
    (A) Direct Linear Equation
    '''

    # the cross product gives us the direction of the line at the intersection
    # of the two planes, and gives us an easy way to check if the two planes
    # are parallel - the cross product will have zero magnitude
    direction = np.cross(p1[:3],p2[:3])
    magnitude = np.sqrt((direction**2).sum())
    if magnitude == 0 :
        return None, None
      

    # now find a point on the intersection. We use the 'Direct Linear Equation'
    # method described in the linked page, and we choose which coordinate
    # to set as zero by seeing which has the largest absolute value in the
    # directional vector

    X = np.abs(direction[0])
    Y = np.abs(direction[1])
    Z = np.abs(direction[2])

    if (Z >= X) & (Z >= Y) :
        point = solveIntersectingPoint(2, 0, 1, p1, p2)
    elif (Y >= Z) & (Y >= X):
        point = solveIntersectingPoint(1, 0, 2, p1, p2)
    else :
        point = solveIntersectingPoint(0, 1, 2, p1, p2)

    return [point, direction]



#####################################################
def solveIntersectingPoint(idxzeroCoord, idxx, idxy, p1, p2):

    '''
    This method helps finding a point on the intersection between two planes.
    Depending on the orientation of the planes, the problem could solve for the
    zero point on either the x, y or z axis
    '''
    a1 = p1[idxx]
    b1 = p1[idxy]
    d1 = p1[3]

    a2 = p2[idxx]
    b2 = p2[idxy]
    d2 = p2[3]

    X0 = ((b2 * d1) - (b1 * d2)) / ((-a1 * b2 + a2 * b1))
    Y0 = ((a1 * d2) - (a2 * d1)) / ((-a1 * b2 + a2 * b1))

    point = np.zeros(3)
    point[idxzeroCoord] = 0
    point[idxx] = X0
    point[idxy] = Y0

    return point



#####################################################
def surface(plan_coeff):

    if plan_coeff[2] != 0:
        xx, yy = np.meshgrid(range(-100,100), range(-100,100))
        # calculate corresponding z
        zz = (-plan_coeff[0] * xx - plan_coeff[1] * yy - plan_coeff[3]) * 1. /plan_coeff[2]
    else:
        if plan_coeff[1] != 0:
            xx, zz = np.meshgrid(range(-100,100), range(-100,100))
            yy = (-plan_coeff[0] * xx - plan_coeff[2] * zz - plan_coeff[3] ) * 1. /plan_coeff[1]
        elif plan_coeff[0] != 0:
            yy, zz = np.meshgrid(range(-100,100), range(-100,100))
            xx = (-plan_coeff[1] * yy - plan_coeff[2] * zz - plan_coeff[3]) * 1. /plan_coeff[0]
        elif (plan_coeff[0] ==0) & (plan_coeff[1]==0):
            return None, None, None

    return xx,yy,zz



#####################################################
if __name__ == '__main__':
#####################################################

    planeA = get_plan_coeff(np.array([.1,.2,.3]),np.array([0,0,0]))
    planeB = get_plan_coeff(np.array([1,.9,-.2]),np.array([2,-19,-3]))

    point, direction = intersectPlanes(planeA, planeB)

    if (point*planeA[:3]).sum()+planeA[3] != 0: 
        print 'point not on plane A'
    if (point*planeB[:3]).sum()+planeB[3] != 0: 
        print 'point not on plane B'

    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #plane A
    xx, yy, zz = surface(planeA)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='b')
    
    #plane B
    xx, yy, zz = surface(planeB)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='g')

    #point
    ax.scatter(point[0] , point[1] , point[2],  color='r')
    
    #line
    dt = np.linspace(-100,100)
    xline,yline,zline = np.array([point+ dt_*direction for dt_ in dt]).T
    ax.plot3D(xline,yline,zline,'red')
    plt.show()

