#include <iostream>
#include <Eigen/Eigen>

using namespace Eigen;
using namespace std;

int main() {
    // Definizione delle matrici A e b per i tre sistemi
    Matrix2d A1, A2, A3;
    Vector2d b1, b2, b3;

    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        -9.992887623566787e-01, 0.0;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        -8.324762492991313e-01, 0.0;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        -8.320502947645361e-01, 0.0;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    // Risoluzione del primo sistema con PALU decomposition
    PartialPivLU<Matrix2d> lu1(A1);
    Vector2d x1 = lu1.solve(b1);

    // Risoluzione del secondo sistema con PALU decomposition
    PartialPivLU<Matrix2d> lu2(A2);
    Vector2d x2 = lu2.solve(b2);

    // Risoluzione del terzo sistema con PALU decomposition
    PartialPivLU<Matrix2d> lu3(A3);
    Vector2d x3 = lu3.solve(b3);

    // Calcolo errore relativo per ciascun sistema
    double rel_err1 = (A1 * x1 - b1).norm() / b1.norm();
    double rel_err2 = (A2 * x2 - b2).norm() / b2.norm();
    double rel_err3 = (A3 * x3 - b3).norm() / b3.norm();

    cout << "Fattorizzazione PALU" << endl;
    cout << "Errore relativo per il sistema 1: " << rel_err1 << endl;
    cout << "Errore relativo per il sistema 2: " << rel_err2 << endl;
    cout << "Errore relativo per il sistema 3: " << rel_err3 << endl;

    // Risoluzione del primo sistema con QR decomposition
    Vector2d x1_b = A1.colPivHouseholderQr().solve(b1);

    // Risoluzione del secondo sistema con QR decomposition
    Vector2d x2_b = A2.colPivHouseholderQr().solve(b2);

    // Risoluzione del terzo sistema con QR decomposition
    Vector2d x3_b = A3.colPivHouseholderQr().solve(b3);

    // Calcolo errore relativo per ciascun sistema
    double rel_err1_b = (A1 * x1_b - b1).norm() / b1.norm();
    double rel_err2_b = (A2 * x2_b - b2).norm() / b2.norm();
    double rel_err3_b = (A3 * x3_b - b3).norm() / b3.norm();

    cout << "Fattorizzazione QR" << endl;
    cout << "Errore relativo per il sistema 1: " << rel_err1_b << endl;
    cout << "Errore relativo per il sistema 2: " << rel_err2_b << endl;
    cout << "Errore relativo per il sistema 3: " << rel_err3_b << endl;

    return 0;
}
