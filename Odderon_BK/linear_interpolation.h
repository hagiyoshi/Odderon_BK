//This cord is borrowed from "http://tips.hecomi.com/entry/20100710/1278786331".
//Linear inter polation inter_polated_f <- CLerp(vector(xgrid),vector(f))

#include <iostream>
#include <functional>

template <class T, template <class A, class Allocator = std::allocator<A> > class Container = std::vector>
class CLerp : public std::unary_function<T, T> // bind�Ŏg����悤�ɂ��邽��
{
private:
	bool Valid;
	Container<T> X, Y;
	const int N;

public:
	CLerp(Container<T> x, Container<T> y) : Valid(false), X(x), Y(y), N(x.size() - 1)
	{
		// �v�f����2�ȏォ���ׂ�
		if (X.size() < 2 || Y.size() < 2) {
			std::cout << "Error! The size of X or Y must be greater than 2." << std::endl;
			return;
		}

		// �v�f�������������ׂ�
		if (X.size() != Y.size()) {
			std::cout << "Error! The size of X and Y are different." << std::endl;
			return;
		}

		// �P�����������ׂ�
		for (int i = 0; i<N - 1; i++) {
			if (X[i] > X[i + 1]) {
				std::cout << "Error! X must be monotonically increasing." << std::endl;
				return;
			}
		}

		Valid = true;
	}

	T operator()(T x) const {
		// �R���X�g���N�^������łȂ��ꍇ�I��
		if (!Valid) {
			return 0;
		}

		// �ŏ��̗v�f��菬���������ꍇ�C�ŏ���2�̗v�f����`���
		if (x < X[0]) {
			return Y[0] + (Y[1] - Y[0]) / (X[1] - X[0]) * (x - X[0]);
		}
		// �Ō�̗v�f���傫�������ꍇ�C�Ō��2�̗v�f����`���
		if (x > X[N]) {
			return Y[N] + (Y[N] - Y[N - 1]) / (X[N] - X[N - 1]) * (x - X[N]);
		}
		// �͈͓��̏ꍇ
		int cnt = 0, prev, next;
		bool flag = false;
		while (cnt < N) {
			if (x >= X[cnt] && x <= X[cnt + 1]) {
				prev = cnt;
				next = cnt + 1;
				flag = true;
				break;
			}
			cnt++;
		}
		return Y[prev] + (Y[next] - Y[prev]) / (X[next] - X[prev]) * (x - X[prev]);
	}

};
