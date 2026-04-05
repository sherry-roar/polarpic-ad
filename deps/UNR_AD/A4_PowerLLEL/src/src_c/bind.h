#pragma once

#define BIND_EXT(name, dim1, dim2, base1, base2, base3) \
    int name##_dim1 = dim1; \
    int name##_dim2 = dim2; \
    int name##_base1 = base1; \
    int name##_base2 = base2; \
    int name##_base3 = base3;

// Index counts from 1
#define BIND1(name) BIND_EXT(name, 0, 0, 1, 1, 1)
#define BIND1_EXT(name, base1) BIND_EXT(name, 0, 0, base1, 1, 1)
#define BIND2(name, dim1) BIND_EXT(name, dim1, 0, 1, 1, 1)
#define BIND2_EXT(name, dim1, base1, base2) BIND_EXT(name, dim1, 0, base1, base2, 1)
#define BIND3(name, dim1, dim2) BIND_EXT(name, dim1, dim2, 1, 1, 1)
#define BIND3_EXT(name, dim1, dim2, base1, base2, base3) BIND_EXT(name, dim1, dim2, base1, base2, base3)

// Use Fortran column-major order
#define I1(name, x) (name[x - name##_base1])
#define I2(name, x, y) (name[x - name##_base1 + (y - name##_base2) * name##_dim1])
#define I3(name, x, y, z) (name[x - name##_base1 + (y - name##_base2) * name##_dim1 + (z - name##_base3) * name##_dim1 * name##_dim2])

// Fortran style DO loop
#define DO(name, start, end) for (int name = start; name <= end; ++name)
