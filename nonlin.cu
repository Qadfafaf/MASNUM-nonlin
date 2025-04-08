__global__ void snonlin_gpu(int nx,int ny,cal_type* awk_array,cal_type cong,cal_type* fconst0_array
        ,cal_type* se_array,cal_type* dse_array,cal_type al13,cal_type al23,cal_type al11,cal_type al21,cal_type al31
        ,int k_length,int j_length,int klp1,int pnts_num,int pnts_num_calc)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy*nx + ix + 1;
    int pnt_id = 1;
    for(pnt_id = idx;pnt_id <= pnts_num_calc; pnt_id+=nx*ny)
    {
        cal_type depth=d_array[pnt_id-1];
        cal_type awk_value=awk_array[pnt_id-1];
        cal_type xx = 0.75*depth*awk_value;
        xx = xx>0.5?xx:0.5;
        cal_type enh = 1+(5.5/xx)*(1-0.833*xx)*exp(-1.25*xx);
        int kh=0;
        int k_iter=1,mr_iter=1,j_iter=1;
        for(k_iter=1;k_iter<=k_length;k_iter++)
        {
            cal_type wp11=wp_array[0*2*k_length+0*k_length+k_iter-1];
            cal_type wp12=wp_array[1*2*k_length+0*k_length+k_iter-1];
            cal_type wp21=wp_array[0*2*k_length+1*k_length+k_iter-1];
            cal_type wp22=wp_array[1*2*k_length+1*k_length+k_iter-1];
            cal_type wm11=wm_array[0*2*k_length+0*k_length+k_iter-1];
            cal_type wm12=wm_array[1*2*k_length+0*k_length+k_iter-1];
            cal_type wm21=wm_array[0*2*k_length+1*k_length+k_iter-1];
            cal_type wm22=wm_array[1*2*k_length+1*k_length+k_iter-1];
            int ip = ikp_array[k_iter-1];
            int ip1 = ikp1_array[k_iter-1];
            int im = ikm_array[k_iter-1];
            int im1 = ikm1_array[k_iter-1];
            int kp = ip;
            int kp1 = ip1;
            int kp2 = ip;
            int kp3 = ip1;
            cal_type ffacp = 1.0;
            cal_type ffacp1 = 1.0;
            cal_type cwks17 = cong * wks17_array[k_iter-1];
            if(kp>=k_length)
            {
                kh+=1;
                kp2=k_length+1;
                if(kp==k_length)
                {
                    kp2=k_length;
                }
                kp=k_length;
                kp1=k_length;
                kp3=k_length+1;
                ffacp = wkh_array[kh-1];
                ffacp1 = wkh_array[kh];
            }
            for(mr_iter=1;mr_iter<=2;mr_iter++)
            {
                for(j_iter=1;j_iter<=j_length;j_iter++)
                {
                    int j11 = jp1_array[(j_iter-1)*2+mr_iter-1];
                    int j12 = jp2_array[(j_iter-1)*2+mr_iter-1];
                    int j21 = jm1_array[(j_iter-1)*2+mr_iter-1];
                    int j22 = jm2_array[(j_iter-1)*2+mr_iter-1];
                    cal_type e_value = e_array[(pnt_id)*j_length*k_length+(j_iter-1)*k_length+(k_iter-1)];
                    if(e_value < 1e-20)
                        continue;
                    cal_type ea1 = e_array[(pnt_id)*j_length*k_length+(j11-1)*k_length+kp-1];
                    cal_type ea2 = e_array[(pnt_id)*j_length*k_length+(j12-1)*k_length+kp-1];
                    cal_type ea3 = e_array[(pnt_id)*j_length*k_length+(j11-1)*k_length+kp1-1];
                    cal_type ea4 = e_array[(pnt_id)*j_length*k_length+(j12-1)*k_length+kp1-1];
                    cal_type ea5 = e_array[(pnt_id)*j_length*k_length+(j21-1)*k_length+im-1];
                    cal_type ea6 = e_array[(pnt_id)*j_length*k_length+(j22-1)*k_length+im-1];
                    cal_type ea7 = e_array[(pnt_id)*j_length*k_length+(j21-1)*k_length+im1-1];
                    cal_type ea8 = e_array[(pnt_id)*j_length*k_length+(j22-1)*k_length+im1-1];

                    cal_type up = (wp11*ea1+wp12*ea2)*ffacp;
                    cal_type up1 = (wp21*ea3+wp22*ea4)*ffacp1;
                    cal_type um = wm11*ea5+wm12*ea6;
                    cal_type um1 = wm21*ea7+wm22*ea8;
                    cal_type sap = up+up1;
                    cal_type sam = um+um1;
                    cal_type e_square = pow(e_value,2);
                    cal_type zua=2.0*e_value/al31;
                    cal_type ead1=sap/al11+sam/al21;
                    cal_type ead2=-2.0*sap*sam/al31;
                    cal_type fcen=fconst0_array[(pnt_id-1)*k_length+k_iter-1]*enh;
                    cal_type ad=cwks17*(e_square*ead1+ead2*e_value)*fcen;
                    cal_type adp=ad/al13;
                    cal_type adm=ad/al23;
                    cal_type delad=cwks17*(e_value*2*ead1+ead2)*fcen;
                    cal_type deladp=cwks17*(e_square/al11-zua*sam)*fcen/al13;
                    cal_type deladm=cwks17*(e_square/al21-zua*sap)*fcen/al23;
                    se_array[(pnt_id-1)*klp1*j_length+(j_iter-1)*klp1+k_iter-1] -= 2*ad;
                    se_array[(pnt_id-1)*klp1*j_length+(j11-1)*klp1+kp2-1] += adp * wp11;
                    se_array[(pnt_id-1)*klp1*j_length+(j12-1)*klp1+kp2-1] += adp * wp12;
                    se_array[(pnt_id-1)*klp1*j_length+(j11-1)*klp1+kp3-1] += adp * wp21;
                    se_array[(pnt_id-1)*klp1*j_length+(j12-1)*klp1+kp3-1] += adp * wp22;
                    se_array[(pnt_id-1)*klp1*j_length+(j21-1)*klp1+im-1] += adm * wm11;
                    se_array[(pnt_id-1)*klp1*j_length+(j22-1)*klp1+im-1] += adm * wm12;
                    se_array[(pnt_id-1)*klp1*j_length+(j21-1)*klp1+im1-1] += adm * wm21;
                    se_array[(pnt_id-1)*klp1*j_length+(j22-1)*klp1+im1-1] += adm * wm22;
                    dse_array[(pnt_id-1)*klp1*j_length+(j_iter-1)*klp1+k_iter-1] -= 2*delad;
                    dse_array[(pnt_id-1)*klp1*j_length+(j11-1)*klp1+kp2-1] += deladp * pow(wp11,2);
                    dse_array[(pnt_id-1)*klp1*j_length+(j12-1)*klp1+kp2-1] += deladp * pow(wp12,2);
                    dse_array[(pnt_id-1)*klp1*j_length+(j11-1)*klp1+kp3-1] += deladp * pow(wp21,2);
                    dse_array[(pnt_id-1)*klp1*j_length+(j12-1)*klp1+kp3-1] += deladp * pow(wp22,2);
                    dse_array[(pnt_id-1)*klp1*j_length+(j21-1)*klp1+im-1] += deladm * pow(wm11,2);
                    dse_array[(pnt_id-1)*klp1*j_length+(j22-1)*klp1+im-1] += deladm * pow(wm12,2);
                    dse_array[(pnt_id-1)*klp1*j_length+(j21-1)*klp1+im1-1] += deladm * pow(wm21,2);
                    dse_array[(pnt_id-1)*klp1*j_length+(j22-1)*klp1+im1-1] += deladm * pow(wm22,2);
                }
            }
        }
    }
}