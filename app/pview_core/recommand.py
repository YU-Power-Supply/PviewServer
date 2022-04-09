from app.common.consts import A_CRITERIA, B_CRITERIA, C_CRITERIA, D_CRITERIA, \
                              E_CRITERIA, E_CRITERIA, F_CRITERIA, G_CRITERIA

'''
#tester
A_CRITERIA = [0.25, 0.5, 0.75, 1.0]
B_CRITERIA = [0.3, 0.7, 1.0]
C_CRITERIA = [0.2, 0.5, 1.0]
D_CRITERIA = [0.5, 1.0]
E_CRITERIA = [0.5, 1.0]

F_CRITERIA = [0.2, 0.4, 0.6, 0.8, 1.0]
G_CRITERIA = [0.2, 0.4, 0.6, 0.8, 1.0]

'''

def recommand(oilly, pih, wrinkle, skin_tone, dead_skin, pore_detect):

    A = A_acne(pore_detect, oilly, A_CRITERIA)
    B = B_stimulus(pih, B_CRITERIA)  # 임시로 100으로 나눔
    C = C_whitening(skin_tone/6, C_CRITERIA)  # 임시로 백분율 변경
    D = D_wrinkle(wrinkle, D_CRITERIA)
    E = E_moisture(dead_skin, E_CRITERIA)

    F = F_moisturizing(oilly, dead_skin, F_CRITERIA)
    G = G_oil(oilly, G_CRITERIA)
    print(A, B, C, D, E, F, G)
    return dict(a_acne=A, a_stimulus=B, a_whitening=C, a_wrinkle=D, a_moisture=E, s_moisturizing=F, s_oil=G)


def A_acne(pore_detect, oilly, A_CRITERIA):
    type_score = pore_detect * 0.5 + oilly * 0.5  # 모공점수 50%, 유분점수 50%
    print(type_score, A_CRITERIA)
#if type_score > 1:
#		return len(A_CRITERIA)-1
    for criteria in A_CRITERIA:
        if type_score <= criteria:
            return A_CRITERIA.index(criteria)


def B_stimulus(pih, B_CRITERIA):  # 색소침착 점수 100%
    print(pih, B_CRITERIA)
#	if pih > 1:
#		return len(B_CRITERIA)-1
    for criteria in B_CRITERIA:
        if pih <= criteria:
            return B_CRITERIA.index(criteria)


def C_whitening(skin_tone, C_CRITERIA):  # 피부톤 점수 100%
    print(skin_tone, C_CRITERIA)
#	if skin_tone > 1:
#		return len(C_CRITERIA)-1
    for criteria in C_CRITERIA:
        if skin_tone <= criteria:
            return C_CRITERIA.index(criteria)


def D_wrinkle(wrinkle, D_CRITERIA):  # 주름 점수 100%
    print(wrinkle, D_CRITERIA)
#	if wrinkle > 1:
#		return len(D_CRITERIA)-1
    for criteria in D_CRITERIA:
        if wrinkle <= criteria:
            return D_CRITERIA.index(criteria)


def E_moisture(dead_skin, E_CRITERIA):  # 각질 점수 100%
    print(dead_skin, E_CRITERIA)
#   if criteria > 1:
#		return len(E_CRITERIA)-1
    for criteria in E_CRITERIA:
        if dead_skin <= criteria:
            return E_CRITERIA.index(criteria)


def F_moisturizing(oilly, dead_skin, F_CRITERIA):  # 유분 점수 50% + 각질 점수 50%
    type_score = oilly*0.5 + dead_skin*0.5
#	if type_score > 1:
#		return len(F_CRITERIA)-1
    print(type_score, F_CRITERIA)
    for criteria in F_CRITERIA:
        if type_score <= criteria:
            return F_CRITERIA.index(criteria)


def G_oil(oilly, G_CRITERIA):  # 유분 점수 100%
    print(oilly, G_CRITERIA)
#	if oilly > 1:
#		return len(G_CRITERIA)-1
    for criteria in G_CRITERIA:
        if oilly <= criteria:
            return G_CRITERIA.index(criteria)
