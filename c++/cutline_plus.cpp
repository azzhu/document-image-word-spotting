// cutline_plus.cpp : �������̨Ӧ�ó������ڵ㡣
//

//#include "stdafx.h"
#include <iostream>
//#include <io.h>
#include <time.h>
#include <fstream>
//#include<direct.h>
//#include <algorithm>
#include <numeric>

#include <opencv2/opencv.hpp>
//#include <d:/myfile/zqjtools.hpp>

using namespace cv;
using namespace std;

#ifdef __cplusplus
extern "C"
{
#endif

//ȫ�ֱ������о࣬�и�
int HJ = -1;
int KD = -1;
//# ͼ��ߺͿ�
int W = 1;
int H = 1;
// heatmap
Mat IMG_HM = Mat::zeros(H, W, CV_8UC1);
//������к�ÿһ�е������꣨rect���ϵ㣩��Ϣ������������
vector<pair<int, int> > MSPOS;
//debug mode switch
bool DEBUG = false;

//��������С����ƽ���ı��Σ����ĸ����㣬�Ա��������任
class WarpPs
{
public:
	WarpPs(Point p11, Point p22, Point p33, Point p44)
	{
		p1 = p11;
		p2 = p22;
		p3 = p33;
		p4 = p44;
	}
	//����Ϊ���ϡ����ϡ����¡����µ㣻
	Point p1;
	Point p2;
	Point p3;
	Point p4;
};

//resize
void src_resize(Mat &src)
{
	int r0 = src.rows;
	int c0 = src.cols;
	if (r0 > c0&&r0 > 900)
	{
		float ra = float(r0) / float(c0);
		resize(src, src, Size(int(900 / ra), 900));
	}
	else if (c0 >= r0&&c0 > 900)
	{
		float ra = float(c0) / float(r0);
		resize(src, src, Size(900, int(900 / ra)));
	}

	if (DEBUG)
		cout << "src_size:" << r0 << "," << c0 << endl;
	return;
}

#define DIR0 1.57075
//# ������д���������ֵͼ��
float calDir(Mat* imgbin)
{
	Mat img_bin;
	imgbin->copyTo(img_bin);
	Mat edges;
	Canny(img_bin, edges, 200, 100);
	int param = 100;
	vector<Vec2f> ls;
	while (true)
	{
		ls.clear();
		HoughLines(edges, ls, 1, CV_PI / 180, param);
		int ls_s = ls.size();
		if (ls_s <= 5)
			param -= 5;
		else if (ls_s >= 500)
			param += 5;
		else
			break;
		if (param <= 0 || param >= 300)
		{
			if (DEBUG)
				cout << "Dir: " << DIR0 << endl;
			return DIR0;
		}
	}

	int ls_s = ls.size();
	vector<float> roate_list;
	for (int i = 0; i < ls_s; i++)
	{
		if (abs(ls[i][1] - DIR0) < 0.2)
			roate_list.push_back(ls[i][1]);
	}
	int rl_s = roate_list.size();
	float roate;
	if (rl_s > 0)
		roate = accumulate(roate_list.begin(), roate_list.end(), 0.0) / rl_s;
	else
		roate = DIR0;
	if (DEBUG)
		cout << "Dir:" << roate << endl;
	return roate;
}

//# ������д������תͼ��
Mat warpImg(Mat* img, float dir)
{
	if (abs(dir - DIR0) < 0.05 || abs(dir - DIR0) > 0.5)
	{
		//# �Ƕȹ�Сʱ����Ҫ��ת��������ʱ�����ϳ������п����Ǹ�������
		//# ��ɵķ���������Ҳ���账��
		if (DEBUG)
			cout << "warpImg:NOT" << endl;
		return *img;
	}

	int r = img->rows;
	int c = img->cols;
	float theta = 3.1416 - dir;
	Point2f srcTri[3];
	Point2f dstTri[3];
	Mat warp_mat(2, 3, CV_32FC1);
	srcTri[0] = Point(int(c / 2), int(r / 2));
	srcTri[1] = Point(int(c / 2), int(r - 1));
	srcTri[2] = Point(int(c - 1), int(r / 2));
	dstTri[0] = Point(int(c / 2), int(r / 2));
	dstTri[1] = Point(int(c / 2 + (r / 2) * cos(theta)), int(r / 2 + (r / 2) * sin(theta)));
	dstTri[2] = Point(int(c / 2 + (c / 2) * sin(theta)), int(r / 2 - (c / 2) * cos(theta)));
	warp_mat = getAffineTransform(srcTri, dstTri);
	Mat dst;
	warpAffine(*img, dst, warp_mat, img->size(), cv::WARP_FILL_OUTLIERS, cv::BORDER_WRAP);

	if (DEBUG)
		cout << "warpImg:YES" << endl;
	return dst;
}

Mat get_sumimg(Mat img_heat, int hw, int dw)
{
	//img_heat��01��ֵͼ��hw��dw�ֱ�Ϊ�˸ߺ˿�

	int hight = img_heat.rows;
	int width = img_heat.cols;

	//pΪ����ͼ
	Mat p = Mat::zeros(hight + 1, width + 1, CV_32SC1);
	img_heat.copyTo(p(Rect(1, 1, width, hight)));
	for (int i = 1; i < hight + 1; i++)
		p(Rect(0, i, width + 1, 1)) += p(Rect(0, i - 1, width + 1, 1));
	for (int i = 1; i < width + 1; i++)
		p(Rect(i, 0, 1, hight + 1)) += p(Rect(i - 1, 0, 1, hight + 1));

	Rect r1, r2, r3, r4;
	r1 = Rect(dw, hw, width + 1 - dw, hight + 1 - hw);
	r2 = Rect(0, 0, width + 1 - dw, hight + 1 - hw);
	r3 = Rect(dw, 0, width + 1 - dw, hight + 1 - hw);
	r4 = Rect(0, hw, width + 1 - dw, hight + 1 - hw);
	//ansΪ������������
	Mat ans = p(r1) + p(r2) - p(r3) - p(r4);

	return ans;
}

//����heatmap�����ִ��ڵ�����,heatmap�ص㣺�����²����У�����������
//�����ֵͼ��
Mat get_heatmap_v3(Mat& img0)
{
	//param
	float th = 0.14;
	int ker = 40;	//������չ��Χ
	int h = 1;	//roiС����߶ȵ�һ�룬������չ��Χ

	int th_num = (2 * ker + 1)*(2 * h + 1)*th;

	//ԭͼ��pad
	int c0 = img0.cols, r0 = img0.rows;
	Mat img = Mat::ones(r0 + h * 2, c0 + ker * 2, 0);
	img *= 255;
	img0.copyTo(Mat(img, Range(h, h + r0), Range(ker, ker + c0)));

	int c = img.cols, r = img.rows;
	Mat heatmap(r, c, CV_8UC1);
	heatmap.setTo(0);

	Mat img_heat = img / 255;
	int hw = 2 * h + 1;
	int dw = 2 * ker + 1;
	Mat grad = get_sumimg(img_heat, hw, dw);

	Mat heatval = ((2 * ker + 1) * (2 * h + 1) - grad) > th_num;
	heatval *= 255;
	heatval.copyTo(heatmap(Rect(ker, h, c - dw + 1, r - hw + 1)));

	medianBlur(heatmap, heatmap, 5);
	dilate(heatmap, heatmap, getStructuringElement(MORPH_RECT, Size(5, 5)));

	return Mat(heatmap, Range(h, h + r0), Range(ker, ker + c0));
}

//�Ż�heatmap���ٶȺܿ�
void heatmap_opt(Mat* heatmap)
{
	/*
	˼�룺���һ�����Ǻڵģ�������������th����Ҳ�Ǻڵģ�
	������������м����е㶼�úڡ�
	*/

	//prarm
	int th = 10;	//���ڸ߶�

	int r = heatmap->rows;
	int c = heatmap->cols;

	//Ϊ��Python�����������һ��(���ﴰ�ڸ߶���ʵֻ��th-1)��th�Լ�1
	++th;
	//���������������������������жϰ׵��ٴ���
	for (int i = 1; i < r; ++i)
	{
		uchar* p = heatmap->ptr<uchar>(i);
		if (i > r - th + 1)
		{
			for (int j = 0; j < c; ++j)
			{
				if (*p == 255)	*p = 0;
				++p;
			}
		}
		else
		{
			for (int j = 0; j < c; ++j)
			{
				if (*p == 255 &&
					*(p - heatmap->step) == 0)
				{	//���±���th-2���㣬�Һڵĵ㣬�ҵ��Ͱ�֮ǰ�ĵ�ȫ���úڣ��Ҳ���������
					for (int l = 1; l < th - 1; ++l)
						if (*(p + heatmap->step*l) == 0)
							for (int k = 0; k < l; ++k)
								*(p + heatmap->step*k) = 0;
				}
				++p;
			}
		}
	}
}

void heatmap_opt_new(Mat* heatmap)
{//�����ԣ��ٶ�û֮ǰ��ѭ����,���Ҳ����
	/*
	˼�룺���һ�����Ǻڵģ�������������th����Ҳ�Ǻڵģ�
	������������м����е㶼�úڡ�
	*/

	//prarm
	int th = 10;	//���ڸ߶�

	int r = heatmap->rows;
	int c = heatmap->cols;
	if (r < th + 2) return;
	*heatmap = *heatmap&Mat::ones(heatmap->size(), heatmap->type());

	Mat imgpad = Mat::zeros(r + 2, c, CV_8UC1);
	heatmap->copyTo(imgpad(Rect(0, 1, c, r)));
	imgpad(Rect(0, 0, c, r + 2 - th)) += imgpad(Rect(0, th, c, r + 2 - th));
	imgpad = imgpad > 0;
	imgpad = imgpad&Mat::ones(imgpad.size(), imgpad.type());

	for (int t = 0; t < th - 1; t++)
	{
		for (int i = r + 1; i > 0; i--)
		{
			imgpad.row(i) &= imgpad.row(i - 1);
			////Mat tem(imgpad.row(i)&imgpad.row(i - 1));
			//tem.copyTo(imgpad.row(i));
		}
	}

	*heatmap = (imgpad(Rect(0, 1, c, r))&*heatmap) * 255;
}

//ȥ��С������,���ٶ��������Ż���
void heatmap_opt2(Mat* heatmap)
{
	//param
	int th = 600;	//��ͨ���������ֵ

	//*heatmap = quxiaodong(heatmap, th);

	cv::Mat labelImage(heatmap->size(), CV_32S);
	cv::Mat stats, centroids;
	int nLabels = connectedComponentsWithStats(*heatmap, labelImage, stats, centroids);

	heatmap->setTo(0);
	for (int i = 1; i < nLabels; ++i)	//nLabels[0]������������
	{
		if (stats.at<int>(i, cv::CC_STAT_AREA) >= th)
			*heatmap += labelImage == i;
	}
}

//�����о���п��
pair<int, int> get_hangnju_kuandu_v2(Mat& heatmap)
{
	Mat src;
	transpose(heatmap, src);
	int c = src.cols, r = src.rows;
	vector<int> hj, kd;

	//��ʮ�����ߣ��ҽ����
	int bu = r / 10;
	for (int k = 1; k < 10; k++)
	{
		int pos = bu*k;
		uchar* p = src.ptr<uchar>(pos);
		p++;
		int st = -1, ed = -1;
		for (int i = 1; i < c; i++)
		{
			int v0 = *(p - 1), v1 = *p;
			if (v0 == 0 && v1 == 255)
			{
				if (st > 0 && ed > 0)
					if (i - st >= 5 && ed - st >= 5 && ed - st < 60)
					{
						hj.push_back(i - st);
						kd.push_back(ed - st);
					}
				st = i;
			}
			else if (v0 == 255 && v1 == 0)
				ed = i;
			p++;
		}
	}

	//���򣬼���ͷ����ƽ��
	sort(hj.begin(), hj.end());
	sort(kd.begin(), kd.end());
	int size = hj.size();
	int tem = size / 7;
	if (tem == 0)	tem = 1;
	int sum_hj = 0, sum_kd = 0;
	for (int i = tem; i < size - tem; i++)
	{
		sum_hj += hj[i];
		sum_kd += kd[i];
	}
	pair<int, int> res;
	if (int(size - tem * 2) == 0)
		res.first = 50;
	else
		res.first = int(sum_hj / (size - tem * 2));
	if (int(size - tem * 2) == 0)
		res.second = 20;
	else
		res.second = int(sum_kd / (size - tem * 2));

	if (res.first < 15 && res.second < 10)
	{
		res.first = 50;
		res.second = 20;
	}

	return res;
}

////��ȡ�����㣬�û����ߵķ�����
vector<vector<Point> > get_featurePs_V2(Mat* heatmap)
{
	//Ϊ���ڴ����ȶ�ͼ�����ת��
	Mat img;
	transpose(*heatmap, img);

	int c = img.cols, r = img.rows;

	//�������ܼ��ȣ�������������
	int ls = 19;
	int stp = r / (ls + 1);

	//���濪ʼ������ÿ������ȡ������
	vector<vector<Point> > fps;
	vector<Point> temp;
	for (int l = 1; l < ls + 1; l++)
	{
		int pos = stp*l;
		uchar* p = img.ptr<uchar>(pos);
		p++;
		temp.clear();
		int st = -1, ed = -1;
		for (int i = 1; i < c; i++)
		{
			int v0 = *(p - 1), v1 = *p;
			if (v0 == 0 && v1 == 255)	//���
			{
				st = i;
			}
			else if (v0 == 255 && v1 == 0)	//�յ�
			{
				ed = i;
				if (st > 0 && ed > 0 && ed - st >= 10)
				{
					//��Ϊ��������ת�ú�ľ������Դ�ʱҪ�ѵ㷴����pushback
					if (ed - st > HJ)
					{
						int ls = (ed - st) / HJ;	//��ȣ�����
						for (int m = 0; m <= ls; m++)
						{
							if (int(st + KD / 2 + HJ*m) < ed)
							{
								Point pp = Point(pos, int(st + KD / 2 + HJ*m));
								if (heatmap->at<uchar>(pp) == 255)
									temp.push_back(pp);
							}
						}
					}
					else
					{
						temp.push_back(Point(pos, int((st + ed) / 2)));
					}

					int tes = temp.size();
					if (tes >= 2)
					{
						int y1 = temp[tes - 1].y;
						int y2 = temp[tes - 2].y;
						if (y1 - y2 < KD)
						{
							temp.pop_back();
							temp.pop_back();
							temp.push_back(Point(pos, int((y1 + y2) / 2)));
						}
					}
				}
			}
			p++;
		}
		fps.push_back(temp);
	}

	//ȥ���յ�
	vector<vector<Point> >::iterator itb = fps.begin();
	for (; itb != fps.end();)
	{
		if (itb->size() == 0)
			itb = fps.erase(itb);
		else
			itb++;
	}

	return fps;
}

bool mycompare(vector<Point> x1, vector<Point> x2)
{
	return x1[0].y < x2[0].y;
}
//���������㣬������������й��࣬����src������һ��src.cols��������Ϣ��û��
vector<vector<Point> > zl_featurePs(vector<vector<Point> > fps, Mat* src)
{
	/*
	�㷨˼�룬�ȴӵ�һ�еó��ĵ���Ϊ�����е�һ����࣬�����ĵ�ɾ����
	�ٴӵڶ��еó��ĵ���Ϊ�����࣬
	������ÿһ��ĵ�һ��������ֵ����;
	���ȥ��ȥ��һ����������ļ��ϣ�����������
	*/

	//param
	//int th = 10;	//��һ����������ƫ�Χ def:9
	int th = HJ*0.5;
	if (th < 3)	th = 3;

	//����ÿ�������һ�㣬û��һ������Ϊ��-1��-1����
	//���ͣ�pair<pair<Point, Point>����ԡ�, pair<int, int>���ڶ������������>
	vector<vector<pair<pair<Point, Point>, pair<int, int> > > > fps_;
	pair<Point, Point> temp;
	pair<int, int> ind;
	vector<pair<pair<Point, Point>, pair<int, int> > > tem;
	pair<pair<Point, Point>, pair<int, int> > te;
	int s0 = fps.size();
	for (int i = 0; i < s0; i++)	//�ȴ���fps_������Ϊ-1
	{
		int s1 = fps[i].size();
		tem.clear();
		for (int j = 0; j < s1; j++)
		{
			temp.first = fps[i][j];
			temp.second = Point(-1, -1);
			ind.first = -1;
			ind.second = -1;
			te.first = temp;
			te.second = ind;
			tem.push_back(te);
		}
		fps_.push_back(tem);
	}

	for (int i = 0; i < s0 - 1; i++)	//��ʼѰ����һ��
	{
		int s1 = fps[i].size();
		for (int j = 0; j < s1; j++)
		{
			int y = fps[i][j].y;
			for (int k = 1; k < 5; k++)
			{
				if (i + k >= s0)	goto mark1;
				int s3 = fps[i + k].size();
				if (s3 < 2)
				{
					int y_ = fps[i + k][0].y;
					if (abs(y - y_) < th)
					{
						fps_[i][j].first.second = fps[i + k][0];
						fps_[i][j].second.first = i + k;
						fps_[i][j].second.second = 0;
						goto mark1;
					}
				}
				else
				{
					for (int m = 0; m < s3 - 1; ++m)
					{
						int y_ = fps[i + k][m].y;
						int y_n = fps[i + k][m + 1].y;
						if (abs(y - y_) < th && abs(y - y_) < abs(y - y_n))
						{
							fps_[i][j].first.second = fps[i + k][m];
							fps_[i][j].second.first = i + k;
							fps_[i][j].second.second = m;
							goto mark1;
						}
					}
					if (abs(y - fps[i + k][s3 - 1].y) < th)
					{
						fps_[i][j].first.second = fps[i + k][s3 - 1];
						fps_[i][j].second.first = i + k;
						fps_[i][j].second.second = s3 - 1;
						goto mark1;
					}
				}
			}
		mark1:;
		}
	}

	//# 第2.5步（新加），当多个点公用同一个下一点的时候，有可能先从下一行开始连接行，这样可能会出问题
	//# 修复方法：保证一个点只有一个上一点，有多个时，只保留距离最近的。
	//# 1，先找出有多个上一点的点
	vector<Point> **fps_lps = new vector<Point>*[s0];
	for (int i = 0; i < s0; ++i)
	{
		int s_i = fps_[i].size();
		fps_lps[i] = new vector<Point>[s_i];
	}
	for (int i = 0; i < s0; ++i)
	{
		int s_i = fps_[i].size();
		for (int j = 0; j < s_i; ++j)
		{
			int x = fps_[i][j].second.first;
			int y = fps_[i][j].second.second;
			if (x > 0 && y > 0)
				fps_lps[x][y].push_back(Point(i, j));
		}
	}
	//# 2, 删除其他距离不是最近的点
	for (int x = 0; x < s0; ++x)
	{
		int s_i = fps_[x].size();
		for (int y = 0; y < s_i; ++y)
		{
			int l_xy = fps_lps[x][y].size();
			if (l_xy > 1)
			{
				int y0 = fps_[x][y].first.first.y;
				int dis = 10000;
				int xx = 0, yy = 0;
				for (int m = 0; m < l_xy; ++m)
				{
					int i_ = fps_lps[x][y][m].x;
					int j_ = fps_lps[x][y][m].y;
					int ym = fps_[i_][j_].first.first.y;
					if (dis > abs(ym - y0))
					{
						dis = abs(ym - y0);
						xx = i_, yy = j_;
					}
				}
				for (int m = 0; m < l_xy; ++m)
				{
					int i_ = fps_lps[x][y][m].x;
					int j_ = fps_lps[x][y][m].y;
					if (i_ != xx || j_ != yy)
					{
						fps_[i_][j_].first.second = Point(-1, -1);
						fps_[i_][j_].second = make_pair(-1, -1);
					}
				}
			}
		}
	}
	for (int i = 0; i < s0; ++i)
	{
		delete[]fps_lps[i];
	}
	delete[]fps_lps;

	//��������ÿһ�еĵ�
	vector<vector<Point> > lpss;
	vector<Point> lps;
	for (int i = 0; i < s0; i++)
	{
		int s1 = fps_[i].size();
		for (int j = 0; j < s1; j++)
		{
			lps.clear();
			int ii = i, jj = j;
			while (1)
			{
				Point p1(fps_[ii][jj].first.first);
				Point p2(fps_[ii][jj].first.second);
				if (p2 == Point(0, 0))	//����ɾ���ĵ㣬������������һ����
				{
					break;
				}
				else if (p2 == Point(-1, -1))	//��β����
				{
					lps.push_back(p1);
					fps_[ii][jj].first.second = Point(0, 0);	//��0,0��ʾɾ���õ�
					break;
				}
				else	//������ɾ���ĵ�
				{
					lps.push_back(p1);
					fps_[ii][jj].first.second = Point(0, 0);	//��0,0��ʾɾ���õ�
					int ii_ = fps_[ii][jj].second.first;
					int jj_ = fps_[ii][jj].second.second;
					ii = ii_, jj = jj_;
				}
			}
			if (lps.size() != 0)
				lpss.push_back(lps);
		}
	}

	//# ȥ������������е�С�㼯
	sort(lpss.begin(), lpss.end(), mycompare);
	vector<vector<Point> > lpss_;
	int l = lpss.size();
	if (l > 2)
	{
		for (int i = 0; i < l - 2; ++i)
			if (lpss[i].size()>1)
				lpss_.push_back(lpss[i]);
		lpss_.push_back(lpss[l - 2]);
		lpss_.push_back(lpss[l - 1]);
	}
	else
	{
		lpss_ = lpss;
	}

	//ÿһ���������߸�����һ�������㣬�����Խ��Ļ�
	int r = src->rows;
	int c = src->cols;
	int ldjj = c / 20;	//ÿһ����������������֮��ļ��
	vector<vector<Point> > lpss__;
	vector<Point> lps__;
	int lpss_size = lpss_.size();
	for (int i = 0; i < lpss_size; i++)
	{
		lps__.clear();
		if (lpss_[i][0].x - ldjj >= 0)
			lps__.push_back(Point(lpss_[i][0].x - ldjj, lpss_[i][0].y));
		int lps_size = lpss_[i].size();
		for (int j = 0; j < lps_size; j++)
		{
			lps__.push_back(lpss_[i][j]);
		}
		if (lpss_[i][lps_size - 1].x + ldjj < c)
			lps__.push_back(Point(lpss_[i][lps_size - 1].x + ldjj, lpss_[i][lps_size - 1].y));
		lpss__.push_back(lps__);
	}

	return lpss__;
}

//Ϊ����任׼�����ռ����д��任��С������ĵ�
vector<vector<WarpPs> > get_WarpPs(Mat *heatmap, vector<vector<Point> > lpss)
{
	vector<vector<WarpPs> > res;
	vector<WarpPs> v_wps;
	if (lpss.size() == 0)
		return res;

	//param
	float th = 2.0;	//ÿһ���ڼ�������Ŀ���ϷŴ���ٱ�

	int c = heatmap->cols, r = heatmap->rows;
	int lj = c / 10;;	//�о࣬���ݻ�������������֮��ľ���
	for (int i = 0; i < lpss.size(); i++)
		if (lpss[i].size() >= 2)
		{
			lj = lpss[i][1].x - lpss[i][0].x;
			break;
		}

	//pair<int, int> temp = get_hangnju_kuandu_v2(*heatmap);
	int hj = HJ;
	int kd = KD;
	int ker = kd / 2 * th;

	int s0 = lpss.size();
	for (int i = 0; i < s0; i++)
	{
		v_wps.clear();

		//ǰ�����⴦��
		int x1 = lpss[i][0].x - lj < 0 ? 0 : lpss[i][0].x - lj;
		int x2 = lpss[i][0].x;
		int y1 = lpss[i][0].y - ker < 0 ? 0 : lpss[i][0].y - ker;
		int y2 = lpss[i][0].y + ker > r - 1 ? r - 1 : lpss[i][0].y + ker;
		Point p1, p2, p3, p4;
		p1 = Point(x1, y1);
		p2 = Point(x2, y1);
		p3 = Point(x1, y2);
		p4 = Point(x2, y2);
		v_wps.push_back(WarpPs(p1, p2, p3, p4));

		//�м䲿��
		int s1 = lpss[i].size();
		for (int j = 0; j < s1 - 1; j++)
		{
			Point p = lpss[i][j];
			Point p_next = lpss[i][j + 1];
			p1 = p.y - ker < 0 ? Point(p.x, 0) : Point(p.x, p.y - ker);
			p2 = p_next.y - ker < 0 ? Point(p_next.x, 0) : Point(p_next.x, p_next.y - ker);
			p3 = p.y + ker > r - 1 ? Point(p.x, r - 1) : Point(p.x, p.y + ker);
			p4 = p_next.y + ker > r - 1 ? Point(p_next.x, r - 1) : Point(p_next.x, p_next.y + ker);
			v_wps.push_back(WarpPs(p1, p2, p3, p4));
		}

		//������⴦��
		x1 = lpss[i][lpss[i].size() - 1].x;
		x2 = lpss[i][lpss[i].size() - 1].x + lj > c - 1 ? c - 1 : lpss[i][lpss[i].size() - 1].x + lj;
		y1 = lpss[i][lpss[i].size() - 1].y - ker < 0 ? 0 : lpss[i][lpss[i].size() - 1].y - ker;
		y2 = lpss[i][lpss[i].size() - 1].y + ker > r - 1 ? r - 1 : lpss[i][lpss[i].size() - 1].y + ker;
		p1 = Point(x1, y1);
		p2 = Point(x2, y1);
		p3 = Point(x1, y2);
		p4 = Point(x2, y2);
		v_wps.push_back(WarpPs(p1, p2, p3, p4));

		res.push_back(v_wps);
	}

	return res;
}

//��������С���򣬷���任���ó�ת�����ÿһ��
vector<Mat> warp_imgs(Mat *src, vector<vector<WarpPs> > wps)
{
	//int lc = src->cols;
	//int lr = src->rows;
	//for (int i = 0; i < wps.size(); i++)
	//	for (int j = 0; j < wps[i].size(); j++)
	//	{
	//		int y2 = wps[i][j].p3.y;
	//		int y1 = wps[i][j].p1.y;
	//		lr = y2 - y1 + 1;
	//		if (y2 != src->rows - 1 && y1 != 0)
	//			goto warp_imgs_mark1;
	//	}
	//warp_imgs_mark1:

	int lc_ = src->cols;
	int lr_ = src->rows;
	int lr = 0;
	int lc = lc_;
	int l_wps = wps.size();
	for (int i = 0; i < l_wps; ++i)
	{
		int l_wpsi = wps[i].size();
		for (int j = 0; j < l_wpsi; ++j)
		{
			int y2 = wps[i][j].p3.y;
			int y1 = wps[i][j].p1.y;
			if (y2 - y1 + 1 > lr)
				lr = y2 - y1 + 1;
		}
	}

	vector<Mat> res;
	if (wps.size() == 0)
		return res;

	MSPOS.clear();
	int s0 = wps.size();
	vector<WarpPs> tm;
	for (int i = 0; i < s0; i++)
	{
		Mat lineimg(lr, lc, CV_8UC1);
		lineimg.setTo(255);
		tm = wps[i];
		MSPOS.push_back(make_pair(tm[0].p1.y, tm[0].p3.y));
		int s1 = tm.size();
		for (int j = 0; j < s1; j++)
		{
			Point p1, p2, p3, p4;
			p1 = tm[j].p1, p2 = tm[j].p2, p3 = tm[j].p3, p4 = tm[j].p4;

			Mat smalldst(p3.y - p1.y + 1, p2.x - p1.x + 1, CV_8UC1);
			Point2f srcTri[3];
			Point2f dstTri[3];
			Mat warp_mat(2, 3, CV_32FC1);
			srcTri[0] = p1;
			srcTri[1] = p2;
			srcTri[2] = p3;
			dstTri[0] = Point(0, 0);
			dstTri[1] = Point(p2.x - p1.x, 0);
			dstTri[2] = Point(0, p3.y - p1.y);
			warp_mat = getAffineTransform(srcTri, dstTri);
			//warpAffine(*src, smalldst, warp_mat, smalldst.size(), cv::InterpolationFlags::INTER_LINEAR, cv::BorderTypes::BORDER_CONSTANT);
			warpAffine(*src, smalldst, warp_mat, smalldst.size(), 8, 0);

			if (smalldst.channels() != 1)
				cvtColor(smalldst, smalldst, CV_BGR2GRAY);
			Rect rec = Rect(p1.x, 0, smalldst.cols, smalldst.rows);
			smalldst.copyTo(lineimg(rec));
		}
		res.push_back(lineimg);
		lineimg.release();	//������ͷ�Ȼ���ٴ����Ļ�������ȥ������Mat���������һ��
	}

	return res;
}

//# pos1, pos2:���е����±߽�����
int get_blank_len_v2(int pos1, int pos2)
{
	int r = pos2 - pos1, c = W;
	int res = c - 1;
	Mat hm(Mat(IMG_HM, Range(pos1, pos2 + 1), Range::all()));
	for (int i = 0; i < c; i++)
	{
		Mat colimg(Mat(hm, Range::all(), Range(i, i + 1)));
		int num = countNonZero(colimg);
		if (num > 5)
		{
			res = i;
			break;
		}
	}
	return res;
}

//# ����ms���ж���һ��ǰ���пո�,,new
vector<pair<bool, Mat> > blank_judge_v2(vector<Mat> ms)
{
	//param
	int th = 25;

	vector<pair<bool, Mat> > res;
	vector<int> hl;	//��ÿһ��ǰ��Ŀհ׵ĳ���
	int s0 = ms.size();
	if (s0 == 0)
		return res;
	if (s0 == 1)
	{
		res.push_back(make_pair(true, ms[0]));
		return res;
	}
	int c = ms[0].cols;
	for (int i = 0; i < s0; ++i)
	{
		hl.push_back(get_blank_len_v2(MSPOS[i].first, MSPOS[i].second));
	}

	for (int i = 0; i < s0; ++i)
	{
		if (hl[i] * 2 > c)
			res.push_back(make_pair(true, ms[i]));
		else if (i == 0)
		{
			if (hl[0] - hl[1] >= th)
				res.push_back(make_pair(true, ms[i]));
			else
				res.push_back(make_pair(false, ms[i]));
		}
		else if (i == s0 - 1)
		{
			if (hl[i] - hl[i - 1] >= th)
				res.push_back(make_pair(true, ms[i]));
			else
				res.push_back(make_pair(false, ms[i]));
		}
		else if (hl[i] - hl[i - 1] >= th || hl[i] - hl[i + 1] >= th)
			res.push_back(make_pair(true, ms[i]));
		else
			res.push_back(make_pair(false, ms[i]));
	}
	return res;
}

//�Զ�ֵ��֮���ͼ��ȥ�»���,�������صķ���������
void cleanUnderline(Mat& imgbin)
{
	//param
	int cd = 15;	//Ӧ��Ϊ��������x�᷽���������ٸ�����ֵΪ0ʱ������������ô������ض�ȥ��

	cd = cd - 1;	//Ϊ��Python��������һ��
	int s, e;	//��¼��ɫ������ʼ��
	s = 0; e = 0;
	int r = imgbin.rows, c = imgbin.cols;
	Mat imgbinpad = Mat::ones(r, c + 2, 0);
	imgbin.copyTo(imgbinpad(Rect(1, 0, c, r)));
	for (int i = 0; i < r; ++i)
	{
		uchar* p = imgbinpad.ptr<uchar>(i);
		++p;
		for (int j = 1; j < c + 2; ++j)
		{
			if (*p == 0 && *(p - 1) != 0)
				s = j;
			else if (*p != 0 && *(p - 1) == 0)
			{
				e = j;
				if (e - s >= cd)
					imgbin(Rect(s - 1, i, e - s, 1)) = 255;
			}
			++p;
		}
	}
}

//�Զ�ֵ��֮���ͼ��ȥ�»��ߣ������ҽ����һ��
void cleanUnderline_slow(Mat& imgbin)
{
	imgbin = imgbin != 0;
	imgbin /= 255;
	int h = imgbin.rows, w = imgbin.cols;

	//Ϊ��x�᷽�����ͼ�������������һ��������
	Mat imgbinpad = Mat::zeros(h, w + 1, 0);
	imgbin.copyTo(Mat(imgbinpad, Range::all(), Range(1, w + 1)));
	for (int i = 1; i < w; ++i)
		imgbinpad.col(i) += imgbinpad.col(i - 1);
	Mat integralImg(Mat(imgbinpad, Range::all(), Range(1, w + 1)));

	//param
	int cd = 15;	//Ӧ��Ϊ��������x�᷽���������ٸ�����ֵΪ0ʱ������������ô������ض�ȥ��
	int ker = cd / 2;
	//# coutΪ�ȽϽ����������������cd������Ϊ��ʱ�������������꣬
	//# ���滹Ҫ�ٴ�����������cd / 2������
	Mat cont;
	Mat(Mat(integralImg, Range::all(), Range(cd - 1, w)) - Mat(integralImg, Range::all(), Range(0, w - cd + 1))).copyTo(cont);
	cont = cont != 0;
	cont /= 255;

	//��������cd/2�����أ�ִ��ker��
	int w_c = cont.cols;
	for (int k = 0; k < ker; ++k)
	{
		for (int i = 0; i < w_c - 1; ++i)
			cont.col(i) &= cont.col(i + 1);
		for (int i = w_c - 1; i > 0; --i)
			cont.col(i) &= cont.col(i - 1);
	}
	Mat roi(imgbin, Range::all(), Range(cd - 1 - ker, w - ker));
	roi = roi == 0;
	roi /= 255;

	//# ���ȥ��ͼ���е��»���
	roi &= cont;
	roi = roi == 0;
	roi /= 255;
	imgbin *= 255;
}

void cleanVerticalLine(Mat& imgbin)
{
	Mat imgbin_T;
	transpose(imgbin, imgbin_T);

	//param
	int cd = 30;	//应设为奇数。在x轴方向连续多少个像素值为0时，把这连续这么多的像素都去掉

	cd = cd - 1;	//为与Python版结果保持一致
	int s, e;	//记录黑色像素起始点
	s = 0; e = 0;
	int r = imgbin_T.rows, c = imgbin_T.cols;
	Mat imgbinpad = Mat::ones(r, c + 2, 0);
	imgbin_T.copyTo(imgbinpad(Rect(1, 0, c, r)));
	for (int i = 0; i < r; ++i)
	{
		uchar* p = imgbinpad.ptr<uchar>(i);
		++p;
		for (int j = 1; j < c + 2; ++j)
		{
			if (*p == 0 && *(p - 1) != 0)
				s = j;
			else if (*p != 0 && *(p - 1) == 0)
			{
				e = j;
				if (e - s >= cd)
					imgbin_T(Rect(s - 1, i, e - s, 1)) = 255;
			}
			++p;
		}
	}

	Mat temp;
	transpose(imgbin_T, temp);
	temp.copyTo(imgbin);
}

//����һ��������ϵ�һ��ͼ��
Mat show_mats(vector<pair<bool, Mat> > mats)
{
	int s = mats.size();
	if (s == 0)
	{
		Mat dst(100, 100, CV_8UC1);
		dst.setTo(0);
		return dst;
	}
	int mc = mats[0].second.cols;
	int mr = mats[0].second.rows;
	int gap = 3;
	Mat dst((mr + gap)*s - gap, mc, CV_8UC1);
	dst.setTo(0);

	for (int i = 0; i < s; i++)
	{
		bool have_blk = mats[i].first;
		Mat mat(mats[i].second);
		Rect rec = Rect(0, i*(mr + gap), mc, mr);
		mat.copyTo(dst(rec));
		if (have_blk)
			circle(dst, Point(15, (mr + gap) * i + int(mr / 2)), 3, 0, 3);
	}
	//���ͼ�������Ļ��ʾ���꣬Ҫresizeһ��
	int h = dst.rows, w = dst.cols;
	if (h > 1010)
	{
		int h_ = 1010;
		int w_ = int(1010 * w / h);
		resize(dst, dst, Size(w_, h_));
		putText(dst, "Resized", Point(2, 50), cv::FONT_HERSHEY_SCRIPT_COMPLEX, 2, Scalar(100, 100, 100));
	}

	return dst;
}

//�������ӻ�
void visualization(Mat imgbin_, Mat& heatmap, vector<vector<Point> > fps, vector<vector<Point> > lpss)
{
	Mat imgbin;
	cvtColor(imgbin_, imgbin, CV_GRAY2BGR);
	heatmap = heatmap != 0;

	int r = imgbin.rows, c = imgbin.cols;
	int ch = imgbin.channels();
	for (int i = 0; i < r; ++i)
	{
		//uchar* p = imgbin.ptr<uchar>(i);
		//uchar* ph = heatmap.ptr<uchar>(i);
		for (int j = 0; j < c; ++j)
		{
			//*(p + 2) -= *ph;
			//p += ch;
			imgbin.at<Vec3b>(i, j)[2] -= heatmap.at<uchar>(i, j);
		}
	}

	int fps_s = fps.size();
	for (int i = 0; i < fps_s; ++i)
	{
		int fps_si = fps[i].size();
		for (int j = 0; j < fps_si; ++j)
		{
			circle(imgbin, fps[i][j], 3, Scalar(0, 0, 255), 2);
		}
	}
	//����֮������
	int lpss_s = lpss.size();
	for (int i = 0; i < lpss_s; ++i)
	{
		int lpss_si = lpss[i].size();
		for (int j = 0; j < lpss_si - 1; ++j)
		{
			Point p1(lpss[i][j]);
			Point p2(lpss[i][j + 1]);
			line(imgbin, p1, p2, Scalar(0, 0, 255), 2);
		}
	}
	imshow("__visualization", imgbin);
}

//run
vector<pair<bool, Mat> > run(Mat& img, bool isbin = false)
{
	//Ԥ����
	if (DEBUG)  cout << "===================" << endl;

	double st;
	st = clock();
//	src_resize(img);
	Mat& src(img);
//    Mat src=img;
//	Mat src;
//	img.copyTo(src);

	H = src.rows, W = src.cols;
	if (DEBUG)	cout << "resizeTime:" << clock() - st << endl;

	st = clock();
	Mat gray, bin;
	if (src.channels() != 1)	{
		cvtColor(src, gray, CV_BGR2GRAY);
		adaptiveThreshold(gray, bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 31, 7);
	}
	else
		adaptiveThreshold(src, bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 31, 7);
	if (DEBUG)	cout << "binimgTime:" << clock() - st << endl;

	st = clock();
	cleanUnderline(bin);
	cleanVerticalLine(bin);
	if (DEBUG)	cout << "cleanUnderlineTime:" << clock() - st << endl;

	st = clock();
	float di = calDir(&bin);
	if (DEBUG)	cout << "calDirTime:" << clock() - st << endl;

	st = clock();
	Mat src_warp = warpImg(&src, di);
	bin = warpImg(&bin, di);
	if (DEBUG)	{
		cout << "warpImgTime:" << clock() - st << endl;
		imshow("bin", bin);
	}

	//ִ��
	st = clock();
	Mat heatmap = get_heatmap_v3(bin);
	if (DEBUG)	cout << "get_heatmapTime:" << clock() - st << endl;

	st = clock();
	heatmap_opt(&heatmap);
	heatmap_opt2(&heatmap);
	if (DEBUG)	cout << "heatmap_optTime:" << clock() - st << endl;

	st = clock();
	heatmap.copyTo(IMG_HM);
	pair<int, int> temp = get_hangnju_kuandu_v2(heatmap);
	HJ = temp.first, KD = temp.second;
	if (DEBUG)	cout << "get_hangnju_kuanduTime:" << clock() - st << endl;

	st = clock();
	vector<vector<Point> > fps = get_featurePs_V2(&heatmap);
	if (DEBUG)	cout << "get_featurePsTime:" << clock() - st << endl;

	st = clock();
	vector<vector<Point> > lpss = zl_featurePs(fps, &bin);
	if (DEBUG)	{
		cout << "zl_featurePsTime:" << clock() - st << endl;
		visualization(bin, heatmap, fps, lpss);
	}

	st = clock();
	vector<vector<WarpPs> > wps = get_WarpPs(&heatmap, lpss);
	if (DEBUG)	cout << "get_WarpPsTime:" << clock() - st << endl;

	st = clock();
	vector<Mat> ms;
	if (isbin)	ms = warp_imgs(&bin, wps);
	else	ms = warp_imgs(&src_warp, wps);
	if (DEBUG)	cout << "warp_imgsTime:" << clock() - st << endl;

	st = clock();
	vector<pair<bool, Mat> > ms_blks = blank_judge_v2(ms);
	if (DEBUG)	{
		cout << "blank_judgeTime:" << clock() - st << endl;
		cout << "===================" << endl;
	}

	return ms_blks;
}

/*****************************************�ָ���*******************************************/
//python api,,,uchar* p_:must gray.######################
vector<pair<bool, Mat> > RES;
int HIG,WIT;
void get_info(uchar* p_, uchar* info_, int r_, int c_)
{
//    DEBUG=1;

	Mat img0 = Mat::zeros(r_, c_, 0),img;
	img0.data = p_;
	img0.copyTo(img);
	vector<pair<bool, Mat> > res = run(img);
	RES=res;
	WIT=c_;
    Mat info=Mat::zeros(1,2,0);
    info.data=info_;
    info.at<uchar>(0,0)=res.size();
    if(res.size()>0)
    {
        HIG=res[0].second.rows;
        info.at<uchar>(0,1)=HIG;
    }
}
bool get_data(uchar* p_,int ind)
{
    Mat line(HIG,WIT,CV_8UC1);
    line.data=p_;
    RES[ind].second.copyTo(line);
    line.release();

    return RES[ind].first;
}

#ifdef __cplusplus
}
#endif

int main()
{
    Mat img=imread("binimg/3.bmp",0);
    vector<pair<bool, Mat> > res=run(img);
    Mat dst=show_mats(res);
    imshow("dst",dst);
    waitKey();
}
