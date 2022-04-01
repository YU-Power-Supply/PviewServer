A_criteria = [0.25, 0.5, 0.75, 1.0]
B_criteria = [0.3, 0.7, 1.0]
C_criteria = [0.2, 0.5, 1.0]
D_criteria = [0.5, 1.0]
E_criteria = [0.5, 1.0]

F_criteria = [0.2, 0.4, 0.6, 0.8, 1.0]
G_criteria = [0.2, 0.4, 0.6, 0.8, 1.0]

oil_dict = {"oilly": 0.9, "normal": 0.6, "dry": 0.3}


def recommand(oilly, pih, wrinkle, skin_tone, dead_skin, pore_detect):

    oilly = oil_dict[oilly]
    A = A_acne(pore_detect, oilly, A_criteria)
    B = B_stimulus(pih/1000, B_criteria)  # 임시로 100으로 나눔
    C = C_whitening(skin_tone/6, C_criteria)  # 임시로 백분율 변경
    D = D_wrinkle(wrinkle, D_criteria)
    E = E_moisture(dead_skin, E_criteria)

    F = F_moisturizing(oilly, dead_skin, F_criteria)
    G = G_oil(oilly, G_criteria)
    print(A, B, C, D, E, F, G)
    return dict(acne=A, stimulus=B, whitening=C, wrinkle=D, moisture=E, moisturizing=F, oil=G)


def A_acne(pore_detect, oilly, A_criteria):
    type_score = pore_detect * 0.5 + oilly * 0.5  # 모공점수 50%, 유분점수 50%
    print(type_score, A_criteria)
    for criteria in A_criteria:
        if type_score <= criteria:
            return A_criteria.index(criteria)


def B_stimulus(pih, B_criteria):  # 색소침착 점수 100%
    print(pih, B_criteria)
    for criteria in B_criteria:
        if pih <= criteria:
            return B_criteria.index(criteria)


def C_whitening(skin_tone, C_criteria):  # 피부톤 점수 100%
    print(skin_tone, C_criteria)
    for criteria in C_criteria:
        if skin_tone <= criteria:
            return C_criteria.index(criteria)


def D_wrinkle(wrinkle, D_criteria):  # 주름 점수 100%
    print(wrinkle, D_criteria)
    for criteria in D_criteria:
        if wrinkle <= criteria:
            return D_criteria.index(criteria)


def E_moisture(dead_skin, E_criteria):  # 각질 점수 100%
    print(dead_skin, E_criteria)
    for criteria in E_criteria:
        if dead_skin <= criteria:
            return E_criteria.index(criteria)


def F_moisturizing(oilly, dead_skin, F_criteria):  # 유분 점수 50% + 각질 점수 50%
    type_score = oilly*0.5 + dead_skin*0.5
    print(type_score, F_criteria)
    for criteria in F_criteria:
        if type_score <= criteria:
            return F_criteria.index(criteria)


def G_oil(oilly, G_criteria):  # 유분 점수 100%
    print(oilly, G_criteria)
    for criteria in G_criteria:
        if oilly <= criteria:
            return G_criteria.index(criteria)
