
def get_fixed_point(float_x, m):
        x = float_x
        fixed_x = 0
        f = 0
        while(int(x)<(2**m)):
            if(int(x*2)<(2**m)):
                x = x*2
                f += 1
                continue
            else:
                fixed_x = int(x)
                break
        return fixed_x, f