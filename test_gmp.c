#include <stdio.h>
#include <gmp.h>

int main() {
    mpz_t n;
    mpz_init(n);
    mpz_set_ui(n, 123456789); // Set n to a large number
    gmp_printf("Here's a large number: %Zd\n", n);
    mpz_clear(n);
    return 0;
}
