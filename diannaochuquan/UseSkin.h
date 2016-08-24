

void SkinCrCbDetect(IplImage* src,IplImage* dst);
void  CandidateSkinArea(IplImage* src,CvMemStorage* store,vector<Rect>& candidate_SkinArea);
void ChangeArea( CvRect& candidateRect1, IplImage* image);

void classifier_hand( Mat& image,  map<string, CvSVM*>&  svms_map, int& category );
void LoadSvms(const string& svms_dir, map<string, CvSVM*>&  svms_map);