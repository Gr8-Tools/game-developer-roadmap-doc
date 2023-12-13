# Unity Game Developer Exam Content

## Theoretical Questions

1. **Vector Spaces**:
   What properties must a set equipped with two operations, vector addition and scalar multiplication, have to be considered a vector space, and how does this foundation apply to the movement and positioning of game objects in Unity?

2. **Dot Product**: 
   Explain how the dot product of two vectors can be used to determine the angle between them and the implications of this in game mechanics such as visibility checks or aligning objects to surfaces.

3. **Cross Product**:
   Describe the cross product and its significance in Unity, particularly in calculating normals or implementing right-handed versus left-handed coordinate systems.

4. **Matrices in Games**:
   Discuss how transformation matrices are constructed in Unity and the importance of matrix multiplication order when translating, rotating, and scaling game objects.

5. **Trigonometry in Unity**:
   How is trigonometry utilized for determining object positioning and movement patterns in Unity's 3D space, and provide an example situation in which sine and cosine functions might be particularly useful.

6. **Quaternions**:
   Define quaternions and contrast them with Euler angles in terms of preventing gimbal lock and ensuring smooth rotational transitions between orientations in Unity.

7. **Slerp vs. Lerp**:
   Compare and contrast Slerp (Spherical Linear Interpolation) with Lerp (Linear Interpolation) in the context of rotating objects in Unity. When would you use one over the other?

8. **Inverse Lerp**:
   Explain what Inverse Lerp is and an example use case where it might be more beneficial than the standard Lerp function in Unity.

9. **Polar Coordinates**:
   Define polar coordinates and their conversion process to Cartesian coordinates in the context of Unity. How might polar coordinates be useful when scripting circular movement?

10. **Curves in Animation**:
    Describe how Bezier and Catmull-Rom curves are used for creating smooth paths and animations in Unity. What are the fundamental differences between these two types of curves?

## Algorithm Questions

1. **Vector Normalization**:
   Outline an algorithm for normalizing a vector in Unity, and explain why normalization is an important step before certain calculations, such as when computing directional vectors.

2. **LookAt Function**:
   Describe an algorithm for a custom "LookAt" function in Unity, which orients an object to face another object, detailing the key steps and trigonometric functions involved.

3. **2D Point Inside Polygon Check**:
   Develop an algorithm that determines if a given 2D point is inside a polygon, useful for click detection in strategy games or UI elements. Explain the steps and how it might be optimized for performance.

4. **Catmull-Rom Spline Interpolation**:
   Specify the algorithm steps for implementing a Catmull-Rom spline interpolation for a camera movement system in Unity.

5. **Inverse Kinematics (2 bone system)**:
   Present a step-by-step algorithm for solving a 2-bone inverse kinematics problem, ensuring a character hand reaches a target, as often used in VR or character animation.

## Coding Tasks

1. **Matrix Multiplication**:
   Write a Unity-compatible function for multiplying two matrices representing transformations in 3D space, adhering to the correct matrix multiplication rules.

2. **Quaternion Slerp Implementation**:
   Implement a script that uses Quaternion Slerp to smoothly rotate an object from its initial orientation to a target orientation over time when a user input is detected.

3. **Bezier Curve Rendering**:
   Code a function that takes a list of anchor and control points and renders a Bezier curve in Unity's editor using Gizmos.

4. **Remap Function**:
   Code a universal "Remap" function that maps a value from one range to another, which is highly useful in Unity for changing the scale of input values.

5. **Transform Object using TRS Matrix**:
   Write a script that constructs a TRS matrix given translation, rotation, and scale, and applies it to transform a Unity GameObject.

---

# Answers and Explanations

## Theoretical Answers

1. **Vector Spaces**: A vector space, in the context of game development, provides a framework for manipulating the position and orientation of game objects. For a set to be a vector space, it must satisfy certain axioms: closure under addition and scalar multiplication, associativity, commutativity, the existence of an additive identity (zero vector), and additive inverses for every vector. These rules underpin operations such as moving objects in a scene, blending animations, and resolving forces in physics simulations.

2. **Dot Product**: The dot product returns the cosine of the angle between two vectors multiplied by the magnitudes of the vectors. In Unity, this allows developers to determine whether an object is in front of or behind another (e.g. for backstab detection in a stealth game) and to calculate the projection of one vector onto another for mechanics like shadows or directional lighting.

3. **Cross Product**: The cross product of two vectors yields a third vector that is perpendicular to both, defining the "normal" direction of surfaces. In Unity, it's used for calculating normals for lighting and determining the handedness of a coordinate system, which affects culling and vertex winding order.

4. **Matrices in Games**: Transformation matrices describe the position, orientation, and scale of objects in Unity. Multiplying matrices combines these transformations, with matrix order affecting the outcome: scale, followed by rotation, and then translation (known as Local or Object space transformations). Understanding this order is crucial for correctly manipulating GameObjects.

5. **Trigonometry in Unity**: Trigonometry is utilized to calculate positions and movement paths on a unit circle or sphere, such as for enemy patrols, bullets in a shmup, or camera control. Sine and cosine are essential for scripting periodic movements, like an object bobbing on water or a platform moving back and forth.

6. **Quaternions**: Quaternions are a complex number system that provides a way to represent 3D rotations avoiding the singularity problems of Euler angles (gimbal lock). In Unity, they ensure a smooth interpolation between rotations and provide easy concatenation of multiple rotations without the need for complex trigonometric calculations.

7. **Slerp vs. Lerp**: While Lerp provides a straight path interpolation between two points or orientations, Slerp interpolates along the shortest path on the unit sphere, which creates a more natural rotational movement. Slerp is thus preferred for smooth orbital rotations, while Lerp might be used for more linear interpolations.

8. **Inverse Lerp**: Inverse Lerp maps a value to a normalized range between 0 and 1 based on its position relative to two endpoints. It's useful in Unity for normalizing speeds, progress bars, or determining the percentage of completion of an action within a determined range.

9. **Polar Coordinates**: Polar coordinates represent a point in space with a distance from a central point and an angle. The conversion to Cartesian coordinates uses trigonometry: `x = r*cos(theta)`, `y = r*sin(theta)`. In Unity, polar coordinates simplify scripts for circular movements like orbits, radars, or windmills.

10. **Curves in Animation**: Bezier curves offer control over the shape with control points, while Catmull-Rom curves pass directly through the specified points. In Unity, they are used for smooth paths and transitions, as in camera paths or motion tweens. Bezier curves are more design-friendly, while Catmull-Rom curves provide direct and uniform control.

## Algorithm Answers

1. **Vector Normalization**:
   Normalize a vector by dividing its components by its magnitude—a non-zero length. In Unity, this ensures that the vector's direction is maintained while its length is reduced to 1, which is essential for representing directions without affecting magnitude, such as velocities or unit directional vectors. 

2. **LookAt Function**:
   A custom LookAt could be created by first calculating the direction to the target object by subtracting the current position from the target's position, normalizing it, and then using trigonometry or quaternion functions to rotate the 'up' vector of the object to align with this direction.

3. **2D Point Inside Polygon Check**:
   The algorithm typically casts a ray from the point and counts the number of intersections with the polygon's edges. An odd count implies the point is inside; even means outside. Performance optimization includes bounding-box checks before raycasting to reduce unnecessary calculations.

4. **Catmull-Rom Spline Interpolation**:
   Calculate the interpolated point using the position of four control points to define a segment on the spline. The algorithm weights these points based on the desired position between them, providing a smooth curve that goes through each control point, often used for camera paths or interpolating movement.

5. **Inverse Kinematics (2 bone system)**:
   An IK algorithm for a 2-bone system usually involves using the law of cosines to calculate the angles required for the 'bones' to reach a target point. The angles are then applied to the joints, rotating them to position the end effector (hand, in this case) at the target location.

## Coding Tasks (Skeleton)

1. **Matrix Multiplication**:
   ```csharp
   public static Matrix4x4 MultiplyMatrices(Matrix4x4 a, Matrix4x4 b) {
       // Placeholder for the matrix multiplication implementation.
   }
   ```

2. **Quaternion Slerp Implementation**:
   ```csharp
   void RotateTowardsTarget(Quaternion targetRotation) {
       transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, Time.deltaTime * rotationSpeed);
   }
   ```

3. **Bezier Curve Rendering**:
   ```csharp
   void OnDrawGizmos() {
       IList<Vector3> points = new List<Vector3>(); // Placeholder for anchor and control points
       // Placeholder for the Bezier curve rendering logic using Gizmos.
   }
   ```

4. **Remap Function**:
   ```csharp
   public static float Remap(float value, float from1, float to1, float from2, float to2) {
       // Placeholder for the remap implementation.
   }
   ```

5. **Transform Object using TRS Matrix**:
   ```csharp
   void ApplyTRSMatrix(Vector3 position, Quaternion rotation, Vector3 scale) {
       Matrix4x4 trs = Matrix4x4.TRS(position, rotation, scale);
       // Placeholder for applying the TRS matrix to the object's transform.
   }
   ```

***

# Exam Content for Unity Game Developer Course

## Theoretical Questions (10)

1. Explain the difference between a vector space and a vector. What are the key properties that define a vector space?

A vector is an element of a vector space that is characterized by its magnitude and direction. A vector space is a collection of vectors that satisfies certain properties like closure under addition and scalar multiplication, existence of an additive identity (zero vector), and existence of additive inverses. Some key properties that define a vector space include:

- Closure under addition: The sum of any two vectors in the space is also a vector in the space. 
- Closure under scalar multiplication: The product of a scalar (real number) and a vector in the space is also a vector in the space.
- Commutativity of addition: The order of vectors doesn't matter when adding them.
- Associativity of addition: When adding three or more vectors, the grouping doesn't matter.
- Additive identity: There exists an additive identity element (usually called the zero vector) such that adding this vector to any other vector doesn't change the other vector. 
- Additive inverse: For every vector there exists an additive inverse such that adding the inverse vector undoes the original addition.
- Distributivity: Scalar multiplication distributes over vector addition.

2. What is the difference between a matrix and a vector? Provide an example of each.

A vector is an element of a vector space that has a magnitude and direction and can be represented as a column or row of elements. For example, a 2D position vector could be represented as [1, 2]. 

A matrix is a rectangular array of numbers, expressions or variables arranged in rows and columns. For example, a 2x3 matrix would look like:

[[1, 2, 3]
 [4, 5, 6]]

The main differences are:

- Vectors have a magnitude and direction, matrices do not inherently have direction. 
- Vectors can be added and multiplied by scalars, matrices can be added and multiplied but the operations are different than for vectors.
- Matrices operate on vectors through multiplication, but vectors cannot be multiplied in the same way.

3. What are the different types of matrix operations? Provide an example of each.

The main types of matrix operations are:

- Addition: Adding two matrices of the same dimensions element-wise. 
- Subtraction: Subtracting matrices of the same dimensions element-wise.
- Multiplication: Multiplying a matrix by a scalar or multiplying two matrices where the inner dimensions must be equal. For example:

  Scalar multiplication: 
  2 * [[1,2],[3,4]] = [[2,4],[6,8]]

  Matrix multiplication:
  [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]

- Transpose: Flipping a matrix over its diagonal to swap its rows and columns. For example:

  [[1,2,3]
   [4,5,6]]

  Transposed is:

  [[1,4] 
   [2,5]
   [3,6]]
   
- Inverse: Finding the matrix inverse operation which undoes the original matrix when multiplied. Only possible for square matrices.

4. What is the difference between a homogeneous and inhomogeneous transformation matrix? Provide an example of each.

A homogeneous transformation matrix includes an extra row/column of values that allow it to represent translations as well as rotations and scales. It is a (N+1)x(N+1) matrix where N is the dimension of the vector space.

An inhomogeneous matrix only represents linear transformations like rotations and scales, and is an NxN matrix. 

For example, in 2D:

Homogeneous matrix for rotation and translation:

[[cosθ, -sinθ, x],
 [sinθ, cosθ, y], 
 [0, 0, 1]]

Inhomogeneous matrix for just rotation:  

[[cosθ, -sinθ],
 [sinθ, cosθ]]

The extra row/column of the homogeneous matrix allows it to encode translations as well.

5. What are the main types of interpolation? Provide an example of how each could be used in game development.

The main types of interpolation are:

- Linear interpolation (lerp): Calculates a value between two end points based on a factor t between 0-1. Could be used to smoothly interpolate between positions for object movement.

- Spherical linear interpolation (slerp): Interpolates orientations represented by quaternions. Could be used to smoothly rotate an object from one orientation to another. 

- Hermite interpolation: Interpolates between end points and also considers tangent vectors at the points. Could be used for smooth animation curves.

- Bezier curves: Interpolates between control points using a polynomial equation. Good for smooth curves like projectile trajectories or spline-based levels.

- Catmull-Rom splines: Interpolate between points and also consider tangents to produce smooth curves. Useful for pathfinding and level generation.

In games they are commonly used for animation, movement, rotations, level/path generation.

6. What are the main differences between Euler angles and quaternions for representing rotation?

The main differences are:

- Euler angles use 3 angles (roll, pitch, yaw) to represent rotation while quaternions use 4 parameters (w, x, y, z) where w is the scalar and x,y,z are the vector.

- Euler angles can suffer from gimbal lock where rotations around successive axes cause loss of a degree of freedom. Quaternions do not have this problem. 

- It is easier to interpolate between quaternions than Euler angles since quaternions do not have the gimbal lock problem.

- Euler angles are human intuitive as yaw/pitch/roll but quaternions are not. 

- Multiplying quaternions is more efficient than concatenating successive rotations from Euler angles.

- Euler angles are susceptible to wind-up errors from small successive rotations whereas quaternions are not.

So in summary, quaternions are preferable for games as they avoid problems with gimbal lock and are more efficient for rotations over time.

7. What is a Bezier curve? How are the control points used to define the shape of the curve?

A Bezier curve is a parametric curve used in computer graphics and design. It is defined by a set of control points P0, P1, ..., Pn. 

The shape of the curve is determined by the position of these control points. It is guaranteed to pass through the first and last points P0 and Pn, called the anchor points. The other points pull the curve towards them but do not define its exact position.

The curve at any point is calculated as the weighted average of the control points, with the weights determined by the Bernstein polynomial basis. This means moving interior points deforms the curve but does not split or disconnect it.

Bezier curves are useful in games for smooth curved paths, spline-based levels, projectile trajectories and more due to their intuitive control point based definition and ability to produce smooth curves.

8. What are the advantages and disadvantages of using polar coordinates over Cartesian coordinates in game development? 

Polar coordinates represent a point by its distance from the origin and angle relative to the x-axis.

Advantages:
- More intuitive for representing angles and rotations than Cartesian coordinates 
- Useful for radial distances in circular UI or targeting reticles
- Simplifies calculations for projectile motion or orbits

Disadvantages:
- Converting to/from Cartesian coordinates requires trigonometric functions which are more expensive than linear algebra
- No concept of direction, just positive/negative angles
- Calculus is more complex for polar than Cartesian coordinates
- Most game engines are designed around Cartesian space rather than polar

So in summary polar coordinates are better suited for angular/radial problems while Cartesian works better for linear algebra and most game engine APIs.

9. What are TRS matrices? How are they used to represent transformations like translation, rotation and scale?

TRS matrices stand for Transformation matrices that encode common transformations like Translation, Rotation and Scale. 

A TRS matrix for a 3D point is a 4x4 homogeneous matrix of the form:

[Sx*Rx, 0, 0, Tx]
[0, Sy*Ry, 0, Ty]  
[0, 0, Sz*Rz, Tz]
[0, 0, 0, 1]

Where:
- Sx,Sy,Sz are scale factors along each axis 
- Rx,Ry,Rz are the rotation matrix for each axis
- Tx,Ty,Tz are the translation along each axis

By concatenating these matrix transformations together, complex transformations can be represented and applied to vectors through a single matrix-vector multiplication.

This makes TRS matrices very useful in games for representing the accumulated transformations of objects in a scene in a compact way.

10. What is the difference between linear and spherical coordinate systems? Provide an example of each.

- Linear (Cartesian) coordinates specify a point using its distances (x,y,z) from fixed orthogonal axes. 

- Spherical coordinates represent a point using its radial distance from the origin (r), polar angle from Z axis (θ), and azimuthal angle in the xy plane (φ).

For example, in 3D:

Cartesian: 
Point(1, 2, 3)

Spherical:  
Point(√(1^2 + 2^2 + 3^2), cos^-1(3/√(1^2 + 2^2 + 3^2)), tan^-1(2/1)) 

Linear is more intuitive for vectors and linear algebra. Spherical is useful for problems involving radial distances and solid angles like planet surfaces, projectile motion or particle systems.

Converting between the two systems allows taking advantage of the properties of each for different problem types.

## Algorithm Questions (5) 

1. Pseudocode for linearly interpolating between two values:

```
Function Lerp(a, b, t)
  t = Clamp01(t) 
  return a * (1 - t) + b * t
End Function

Function Clamp01(t)
  if t < 0 then 
    return 0
  else if t > 1 then
    return 1 
  else
    return t
  end if
end Function
```

2. Pseudocode to remap a value from one range to another: 

```
Function Remap(value, originalMin, originalMax, newMin, newMax)

  normalizedValue = (value - originalMin) / (originalMax - originalMin)
  remappedValue = normalizedValue * (newMax - newMin) + newMin

  return remappedValue

End Function
```

3. Pseudocode for spherical linear interpolation between two orientations represented by quaternions:

```
Function Slerp(q1, q2, t)

  t = Clamp01(t)

  theta = ACos(Dot(q1, q2))
  sinTheta = Sin(theta)  

  if sinTheta < 0.001 then
    return Lerp(q1, q2, t)
  end

  ratioA = Sin((1 - t) * theta) / sinTheta
  ratioB = Sin(t * theta) / sinTheta

  return ratioA * q1 + ratioB * q2

End Function
```

4. Pseudocode to calculate the position along a Catmull-Rom spline segment:

```
Function CatmullRom(p0, p1, p2, p3, t)

  t = Clamp01(t)
  t2 = t * t
  t3 = t2 * t

  return 0.5 * (
   (-t3 + 2*t2 - t) * p1 
   + (3*t3 - 5*t2 + 2) * p2
   + (-3*t3 + 4*t2 + t) * p3  
   + (t3 - t2) * p0
  )

End Function
```

5. Pseudocode to multiply a vector by a TRS matrix:

```
Function TransformVector(vector, matrix)

  result = vector

  result.x = result.x * matrix.scale.x + 
             result.y * matrix.shearXY * matrix.scale.x +  
             result.z * matrix.shearXZ * matrix.scale.x +
             matrix.translate.x
             
  result.y = result.x * matrix.shearYX * matrix.scale.y +  
             result.y * matrix.scale.y +
             result.z * matrix.shearYZ * matrix.scale.y + 
             matrix.translate.y

  result.z = result.x * matrix.shearZX * matrix.scale.z +
             result.y * matrix.shearZY * matrix.scale.z +  
             result.z * matrix.scale.z +
             matrix.translate.z
             
  return result

End Function
```

## Coding Tasks (5)

1. Write a C# method to calculate the dot product of two vectors.

```csharp
public static float Dot(Vector3 a, Vector3 b) 
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
```

2. Write a C# method to calculate the cross product of two 3D vectors. 

```csharp 
public static Vector3 Cross(Vector3 a, Vector3 b)
{
  return new Vector3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  );
}
```

3. Write a C# method to interpolate between two vectors using a factor t.

```csharp
public static Vector3 Lerp(Vector3 a, Vector3 b, float t) 
{
  return a * (1 - t) + b * t;
}
```

4. Write a C# method to rotate a vector around an axis by an angle in radians.

```csharp
public static Vector3 RotateAroundAxis(Vector3 v, Vector3 axis, float angle)
{
  axis = axis.normalized;
  float c = Mathf.Cos(angle);
  float s = Mathf.Sin(angle);
  float omc = 1 - c;

  return v * c + axis * (axis.Dot(v)) * omc + axis.Cross(v) * s;
}
``` 

5. Write a C# method to calculate the position along a Catmull-Rom spline segment.

```csharp
public static Vector3 CatmullRom(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t) 
{
  t = Mathf.Clamp01(t);
  float t2 = t * t;
  float t3 = t2 * t;

  return 0.5f * (
    (-t3 + 2f*t2 - t) * p1 + 
    (3f*t3 - 5f*t2 + 2f) * p2 +
    (-3f*t3 + 4f*t2 + t) * p3 + 
    (t3 - t2) * p0
  );
}
```
