def fp(hex_no, b):
    val = int(hex_no, 16)
    return (val) / 2 ** b

print(fp("3C", 4))