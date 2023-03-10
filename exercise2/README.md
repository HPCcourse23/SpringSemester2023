# Advanced High Performance Computing Exercise Sheet 1

## Exercise 1 [Array Arithmetic]
(i) What happens when you add two arrays together?

    When you add two Blitz arrays together using the + operator, corresponding elements in the two arrays are added together.

## Exercise 2 [Indexing, subarrays and slicing]
(i) This program creates an array A and a subarray B. What does line 6 (A=7) do? What the the
print that follows show?

    The line A = 7; assigns the value 7 to all elements in the 3D Blitz++ array A.
    The output will be a 3D array with dimensions 8x8x4, where all elements in the array have a value of 7.

(ii) The subarray is subsequently set to 4 on line 9 (B=4). What, if anything, does this do to A?

    The array B is constructed using constructor, which creates a subarray of the original array A. Therefore, any modifications made to the subarray through B will also modify the corresponding elements in the original array A.
    Therefore B=4 will assign 4 to range of A array (5, 7)x(5, 7)x(0, 2)

## Exercise 3 [Storage Order]
Code in `storage_order.cpp`