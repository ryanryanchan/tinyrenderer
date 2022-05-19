import numpy as np


def obj_load(filename):
    V, Vi = [], []
    with open(filename) as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                V.append([float(x) for x in values[1:4]])
            elif values[0] == 'f':
                Vi.append([int(x) for x in values[1:4]])
    return np.array(V), np.array(Vi)-1


def draw_line(A, B):
    (x0, y0), (x1, y1) = np.array(A).astype(int), np.array(B).astype(int)
    P = []
    steep = False
    if abs(x0-x1) < abs(y0-y1):
        steep, x0, y0, x1, y1 = True, y0, x0, y1, x1
    if x0 > x1:
        x0, x1, y0, y1 = x1, x0, y1, y0
    dx, dy = x1-x0, y1-y0
    y, error2, derror2 = y0, 0, abs(dy)*2

    for x in range(x0, x1+1):
        if steep:
            P.append((y, x))
        else:
            P.append((x, y))
        error2 += derror2
        if error2 > dx:
            if(y1 > y0):
                y += 1
            else:
                y -= 1
            error2 -= dx * 2
    return P


def triangle_lines(face):
    P = []
    for j in range(0, 3):
        v0 = verts[face[j]]
        v1 = verts[face[(j+1) % 3]]

        x0 = (v0[0]*5 + 0.5) * width
        y0 = (v0[1]*5 + 0) * height
        x1 = (v1[0]*5 + 0.5) * width
        y1 = (v1[1]*5 + 0) * height
        P.extend(draw_line((x0, y0), (x1, y1)))
    return P

# takes in coordinates


def baycentric(points, point):
    u = np.cross([points[2][0] - points[0][0],
                  points[1][0] - points[0][0],
                  points[0][0] - point[0]],
                 [points[2][1] - points[0][1],
                  points[1][1] - points[0][1],
                  points[0][1] - point[1]])
    if abs(u[2]) < 1:
        return [-1, 1, 1]
    return [
        1.0 - (u[0] + u[1])/u[2],
        u[1]/u[2],
        u[0]/u[2]]


# face = [[vx1,vy1,vz1],[vx1,vy1,vz1],[vx1,vy1,vz1]]
def triangle(face, color):
    xmin = int(max(0,
                   min(face[0][0],
                       face[1][0],
                       face[2][0])))
    xmax = int(min(width,
                   max(face[0][0],
                       face[1][0],
                       face[2][0])))

    ymin = int(max(0,
                   min(face[0][1],
                       face[1][1],
                       face[2][1])))
    ymax = int(min(height,
                   max(face[0][1],
                       face[1][1],
                       face[2][1])))

    P = []
    for px in range(xmin, xmax):
        for py in range(ymin, ymax):
            bc = baycentric(face, (px, py))
            if(bc[0] < 0 or bc[1] < 0 or bc[2] < 0):
                continue
            pz = 0
            for i in range(3):
                pz += face[i][2] * bc[i]
            if zBuf[px][py] < pz:
                zBuf[px][py] = pz
                P.append((px, py, color))
    return P


if __name__ == '__main__':
    import time
    import PIL.Image

    verts, faces = obj_load("bunny.obj")

    width, height = 1200, 1200
    image = np.zeros((height, width, 3), dtype=np.uint8)
    light = np.array([0, 0, -1])

    verts[:, 0] = (verts[:, 0] * 5 + 0.5) * width
    verts[:, 1] = (verts[:, 1] * 5 + 0) * height
    verts[:, 2] = (verts[:, 2] * 5 + 0) * width

    verts = list(np.int_((verts)))
    zBuf = np.ones((height, width, 1), dtype=float)
    zBuf *= np.NINF
    P = []
    start = time.time()
    for face in faces:
        screen_coords = []
        world_coords = []
        for i in range(3):
            v = verts[face[i]]
            screen_coords.append([v[0], v[1], v[2]])
            world_coords.append(v)
        vec1 = np.subtract(world_coords[2], world_coords[0])
        vec2 = np.subtract(world_coords[1], world_coords[0])
        n = np.cross(vec1, vec2)
        n = n/np.linalg.norm(n)
        intensity = np.dot(n, light)
        if(intensity > 0):
            P.extend(triangle(screen_coords,
                              [256*intensity,
                               256*intensity,
                               256*intensity]))
    for item in P:
        image[item[0], item[1]] = item[2]
    end = time.time()
    print("Rendering time: {}".format(end-start))
    PIL.Image.fromarray(image[:: -1, :, :]).save("bunny.png")

    zMax = np.amax(zBuf)
    zMin = np.amin(zBuf[zBuf != np.amin(zBuf)])

    zBufImg = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(height):
        for y in range(width):
            if np.isneginf(zBuf[x][y]):
                zBufImg[x][y] = [0, 0, 0]
            else:
                intensity = (zBuf[x][y] - zMin) * 128 / (zMax - zMin) + 64
                zBufImg[x][y] = [intensity, intensity, intensity]
    PIL.Image.fromarray(zBufImg[:: -1, :, :]).save("zBuffer.png")
