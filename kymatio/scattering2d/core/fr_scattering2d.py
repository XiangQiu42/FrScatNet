# Authors: Qiu Xiang
# 在这个版本中，我们将散射网络拓展到分数阶
# 简要来看，我们只需要将卷积操作替换成分数阶卷积即可
# 注意到，根据卷积定理，时域（空域）的卷积对应于频域（也就是原作者注释的Fourier域？）的相乘，
# 故而类似于原代码中的做法，我们也将在Fourier域中完成相乘的工作再通过傅里叶变换的反
# 变换来变换回时域（空域）, 在这里我们使用 alpha_1 和 alpha_2 来表示角度,默认为1,
# 即 \theta = \pi/2*\alpha = \pi/2
# For more information, base_frontend.py _doc_class would be helpful~~~(多看注释QAQ~)
# 如果真的有人看这份代码，请原谅我中英夹杂的注释,逃~~~

def fr_scattering2d(x, pad, unpad, backend, J, L, phi, psi, max_order,
                    out_type='array', cot_1=0, cot_2=0):
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    fft = backend.fft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate
    fr_trans_in = backend.fr_trans_in
    fr_trans_out = backend.fr_trans_out

    # Define lists for output.
    out_S_0, out_S_1, out_S_2 = [], [], []

    # Padding the input for FFt?  ie: if input is (3,32,32),
    # then after padding, the shape of input will be (3,48,48,2)
    U_r = pad(x)

    # 加入分数阶之后的变化(Step 1)， 首先，我们需要将 x(t) -> x_bar(t),
    # x_bat(t) = x(t) *exp(j/2)*t^2*cot\theta
    # Note here: we only realize this transform with pytorch(懒癌QAQ~)
    U_r = fr_trans_in(U_r, cot_1=cot_1, cot_2=cot_2)

    U_0_c = fft(U_r, 'C2C')

    # First low pass filter
    U_1_c = cdgmm(U_0_c, phi[0])
    U_1_c = subsample_fourier(U_1_c, k=2 ** J)

    # 加入分数阶之后的变化(Step 2)，最后，我们需要将 W_x(a,b) -> W_x_bar(a,b), that is:
    # W_x_bar(a,b) = W_x(a,b)*exp(-j/2)*b^2*cot\theta
    # 注意到由于我们还需要乘上一个复矩阵，故而我们并不采用上面的 irfft(the original code),
    # 而是先做 ifft, 做第二个矩阵变换后，再取实部

    # S_0 = fft(U_1_c, 'C2R', inverse=True) # the original code

    S_0 = fft(U_1_c, 'C2C', inverse=True)
    S_0 = fr_trans_out(S_0, out_type='S', cot_1=cot_1, cot_2=cot_2)

    S_0 = unpad(S_0)

    out_S_0.append({'coef': S_0,
                    'j': (),
                    'theta': ()})

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        theta1 = psi[n1]['theta']

        # Note here we don't need add additional step 1 since U_0_c has finished step 1
        # for computing fractional convolution with filters['psi']

        U_1_c = cdgmm(U_0_c, psi[n1][0])
        if j1 > 0:
            U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
        U_1_c = fft(U_1_c, 'C2C', inverse=True)

        # add additional step 2 for computing fractional convolution with filters['psi']
        U_1_c = fr_trans_out(U_1_c, out_type='U', cot_1=cot_1, cot_2=cot_2)

        U_1_c = modulus(U_1_c)

        # add additional step 1 for computing fractional convolution with filters['phi']
        U_1_c = fr_trans_in(U_1_c, cot_1=cot_1, cot_2=cot_2)

        U_1_c = fft(U_1_c, 'C2C')

        # Second low pass filter
        S_1_c = cdgmm(U_1_c, phi[j1])
        S_1_c = subsample_fourier(S_1_c, k=2 ** (J - j1))

        # Do NOT Use torch.irfft (which was used in the original code)
        # S_1_r = fft(S_1_c, 'C2R', inverse=True)
        # then add additional step 2 for computing fractional convolution with filters['phi']
        S_1_r = fft(S_1_c, 'C2C', inverse=True)
        S_1_r = fr_trans_out(S_1_r, out_type='S', cot_1=cot_1, cot_2=cot_2)

        S_1_r = unpad(S_1_r)

        out_S_1.append({'coef': S_1_r,
                        'j': (j1,),
                        'theta': (theta1,)})

        if max_order < 2:
            continue
        for n2 in range(len(psi)):
            j2 = psi[n2]['j']
            theta2 = psi[n2]['theta']

            if j2 <= j1:
                continue

            U_2_c = cdgmm(U_1_c, psi[n2][j1])
            U_2_c = subsample_fourier(U_2_c, k=2 ** (j2 - j1))
            U_2_c = fft(U_2_c, 'C2C', inverse=True)

            # # add additional step 2 for computing fractional convolution with filters['psi']
            U_2_c = fr_trans_out(U_2_c, out_type='U', cot_1=cot_1, cot_2=cot_2)

            U_2_c = modulus(U_2_c)

            # # add additional step 1 for computing fractional convolution with filters['phi']
            U_2_c = fr_trans_in(U_2_c, cot_1=cot_1, cot_2=cot_2)

            U_2_c = fft(U_2_c, 'C2C')

            # Third low pass filter
            S_2_c = cdgmm(U_2_c, phi[j2])
            S_2_c = subsample_fourier(S_2_c, k=2 ** (J - j2))

            # S_2_r = fft(S_2_c, 'C2R', inverse=True)
            S_2_r = fft(S_2_c, 'C2C', inverse=True)
            S_2_r = fr_trans_out(S_2_r, out_type='S', cot_1=cot_1, cot_2=cot_2)

            S_2_r = unpad(S_2_r)

            out_S_2.append({'coef': S_2_r,
                            'j': (j1, j2),
                            'theta': (theta1, theta2)})

    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)

    if out_type == 'array':
        out_S = concatenate([x['coef'] for x in out_S])

    return out_S


__all__ = ['fr_scattering2d']
