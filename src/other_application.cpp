#include "other_application.h"


pair<double, double> calcImageOffset(Mat &image_std, Mat &image_offset, double tmpl_divisor) {
    if (image_std.empty()) {
        throw invalid_argument("calcImageOffset() Error: Unable to load image_std or image_std loading error!");
    }

    if (image_offset.empty()) {
        throw invalid_argument("calcImageOffset() Error: Unable to load image_offset or image_offset loading error!");
    }

    if (tmpl_divisor <= 1) {
        throw invalid_argument("calcImageOffset() Error: Parameter tmpl_divisor must be > 1!");
    }

    int std_width = image_std.cols;
    int std_height = image_std.rows;

    // 从标准图像中截取中心模板
    int tmpl_width = (int) (std_width / tmpl_divisor);
    int tmpl_height = (int) (std_height / tmpl_divisor);
    Mat image_tmpl = image_std(
            Rect((std_width - tmpl_width) / 2, (std_height - tmpl_height) / 2, tmpl_width, tmpl_height));

    // 进行模板匹配
    Mat temp;
    matchTemplate(image_offset, image_tmpl, temp, TM_CCOEFF_NORMED);

    // 定位匹配的位置
    Point max_loc;
    minMaxLoc(temp, nullptr, nullptr, nullptr, &max_loc);

    // 使用亚像素级的插值来精确定位模板的中心
    Point2f subpixel_offset = phaseCorrelate(image_tmpl,
                                             image_offset(Rect(max_loc.x, max_loc.y, tmpl_width, tmpl_height)));

    // 返回结果，中心坐标系像素
    pair<double, double> offset_value;
    offset_value.first = ((double) max_loc.x + subpixel_offset.x) - (int) ((std_width - tmpl_width) / 2);
    offset_value.second = (int) ((std_height - tmpl_height) / 2) - ((double) max_loc.y + subpixel_offset.y);

    return offset_value;
}
