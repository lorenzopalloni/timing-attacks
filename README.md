Timing attacks against RSA
==========================

Side-channel attacks exploit physical parameters such as execution time, supply current and electromagnetic emission to retrieve secrets from a system.
A timing attack is a type of side-channel attack that, by measuring time differences in data-dependent operations of a cryptosystem, can expose its secret key.
This report describes two major timing attacks against RSA, that is a public-key cryptosystem that is widespread in computer security applications.

In 1996, Kocher was the first to design a timing attack against cryptosystems based on modular exponentiation, such as RSA [1].
His work pushed enhancements of modular exponentiation implementations.
If adopted, they make the original attack ineffective.

Almost ten years later, in 2005, Brumley and Boneh proved that timing attacks were practical on network servers based on OpenSSL1 , a commonly used library that implements RSA with several improvements [2].
In their experiments, they retrieved whole RSA private keys in time frames of approximately two hours, even through web servers with multiple network switches.
Their work prompted OpenSSL developers, as well as other crypto libraries, to set up the defence technique "blinding" in RSA implementations by default.

### References

* [1] Kocher, P.C., 1996, August. Timing attacks on implementations of Diffie-Hellman, RSA, DSS, and other systems. In Annual International Cryptology Conference (pp. 104-113). Springer, Berlin, Heidelberg.
* [2] Brumley, D. and Boneh, D., 2005. Remote timing attacks are practical. Computer Networks, 48(5), pp.701-716.
* [3] Boreale, M., 2003. Note per il corso di Sicurezza delle Reti.
* [4] Montgomery, P.L., 1985. Modular multiplication without trial division. Mathematics of computation, 44(170), pp.519-521.
* [5] Menezes, A.J., Van Oorschot, P.C. and Vanstone, S.A., 2018. Handbook of applied cryptography. CRC press.
* [6] Schindler, W., 2000, August. A timing attack against RSA with the chinese remainder theorem. In International Workshop on Cryptographic Hardware and Embedded Systems (pp. 109-124). Springer, Berlin, Heidelberg.
* [7] Coppersmith, D., 1997. Small solutions to polynomial equations, and low exponent RSA vulnerabilities. Journal of cryptology, 10(4), pp.233-260.

