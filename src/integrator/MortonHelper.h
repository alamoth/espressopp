#define uint32 unsigned int
#define int32 int

int3 mapIndexToPosition(long index, int sizeX, int sizeY, int sizeZ){
    int3 ret;
    ret.x = index % sizeX;
    // p1 = index % size[0];
    // index /= size[0];
    index /= sizeX;
    ret.y = index % sizeY;
    // p2 = index % size[1];
    // index /= size[1];
    index /= sizeY;
    // p3 = index;
    ret.z = index;
    return ret;
}

// "Insert" a 0 bit after each of the 16 low bits of x
uint32 Part1By1(uint32 x)
{
  x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  return x;
}

// "Insert" two 0 bits after each of the 10 low bits of x
uint32 Part1By2(uint32 x)
{
  x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
  x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  return x;
}

uint32 EncodeMorton2(uint32 x, uint32 y)
{
  return (Part1By1(y) << 1) + Part1By1(x);
}

uint32 EncodeMorton3(uint32 x, uint32 y, uint32 z)
{
  return (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);
}

// Inverse of Part1By1 - "delete" all odd-indexed bits
uint32 Compact1By1(uint32 x)
{
  x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  x = (x ^ (x >>  1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x >>  2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x >>  4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x >>  8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
  return x;
}

// Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
uint32 Compact1By2(uint32 x)
{
  x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
  return x;
}

uint32 DecodeMorton2X(uint32 code)
{
  return Compact1By1(code >> 0);
}

uint32 DecodeMorton2Y(uint32 code)
{
  return Compact1By1(code >> 1);
}

uint32 DecodeMorton3X(uint32 code)
{
  return Compact1By2(code >> 0);
}

uint32 DecodeMorton3Y(uint32 code)
{
  return Compact1By2(code >> 1);
}

uint32 DecodeMorton3Z(uint32 code)
{
  return Compact1By2(code >> 2);
}