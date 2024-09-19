import numpy as np

def print_matrix(matrix):
    print(np.array(matrix))

def define_matrix(prompt):
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))
    matrix = []
    print(prompt)
    for i in range(rows):
        row = list(map(float, input(f"Enter row {i+1}: ").split()))
        matrix.append(row)
    return np.array(matrix)

def solve_linear_system():
    print("Enter the coefficient matrix (A): ")
    A = define_matrix("Matrix A:")
    print("Enter the constants matrix (B): ")
    B = define_matrix("Matrix B:")
    
    try:
        result = np.linalg.solve(A, B)
        print("The solution to the system is: ")
        print_matrix(result)
    except np.linalg.LinAlgError:
        print("System cannot be solved, matrix might be singular.")

def matrix_menu():
    while True:
        print("\nChoose a matrix operation:")
        print("1. Matrix Addition")
        print("2. Matrix Subtraction")
        print("3. Scalar Matrix Multiplication")
        print("4. Elementwise Matrix Multiplication")
        print("5. Matrix Multiplication")
        print("6. Matrix Transpose")
        print("7. Trace of a Matrix")
        print("8. Solve System of Linear Equations")
        print("9. Determinant")
        print("10. Inverse")
        print("11. Eigenvalues and Eigenvectors")
        print("12. Exit")
        
        choice = int(input("Enter your choice: "))
        
        if choice == 1:
            matrix1 = define_matrix("Enter matrix 1:")
            matrix2 = define_matrix("Enter matrix 2:")
            if matrix1.shape != matrix2.shape:
                print("Matrices must have the same dimensions for addition.")
            else:
                result = np.add(matrix1, matrix2)
                print("The result of matrix addition is:")
                print_matrix(result)
        
        elif choice == 2:
            matrix1 = define_matrix("Enter matrix 1:")
            matrix2 = define_matrix("Enter matrix 2:")
            if matrix1.shape != matrix2.shape:
                print("Matrices must have the same dimensions for subtraction.")
            else:
                result = np.subtract(matrix1, matrix2)
                print("The result of matrix subtraction is:")
                print_matrix(result)

        elif choice == 3:
            matrix = define_matrix("Enter the matrix:")
            scalar = float(input("Enter the scalar value: "))
            result = np.multiply(matrix, scalar)
            print(f"The result of scalar multiplication by {scalar} is:")
            print_matrix(result)

        elif choice == 4:
            matrix1 = define_matrix("Enter matrix 1:")
            matrix2 = define_matrix("Enter matrix 2:")
            if matrix1.shape != matrix2.shape:
                print("Matrices must have the same dimensions for element-wise multiplication.")
            else:
                result = np.multiply(matrix1, matrix2)
                print("The result of element-wise multiplication is:")
                print_matrix(result)
        
        elif choice == 5:
            matrix1 = define_matrix("Enter matrix 1:")
            matrix2 = define_matrix("Enter matrix 2:")
            if matrix1.shape[1] != matrix2.shape[0]:
                print("The number of columns in the first matrix must be equal to the number of rows in the second matrix.")
            else:

