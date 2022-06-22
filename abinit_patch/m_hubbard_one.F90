!!****m* ABINIT/m_hubbard_one
!! NAME
!!  m_hubbard_one
!!
!! FUNCTION
!!
!! Solve Anderson model with the density/density Hubbard one approximation
!!
!! COPYRIGHT
!! Copyright (C) 2006-2021 ABINIT group (BAmadon)
!! This file is distributed under the terms of the
!! GNU General Public License, see ~abinit/COPYING
!! or http://www.gnu.org/copyleft/gpl.txt .
!!
!! INPUTS
!!
!! OUTPUT
!!
!! PARENTS
!!
!! CHILDREN
!!
!! SOURCE

#if defined HAVE_CONFIG_H
#include "config.h"
#endif


#include "abi_common.h"

MODULE m_hubbard_one

 use defs_basis
 use m_abicore
 use m_errors
 use m_xmpi

 use m_crystal, only : crystal_t
 use m_green, only : green_type
 use m_self, only : self_type
 use m_oper, only : oper_type
 use m_hu, only : hu_type
 use m_pawang, only : pawang_type
 use m_paw_dmft, only : paw_dmft_type
 
#if defined HAVE_PYTHON_INVOCATION
 use INVOKE_PYTHON
#endif
 use ISO_C_BINDING
 use netCDF

 implicit none

 private

 public :: hubbard_one
!!***

contains

!!****f* m_hubbard_one/hubbard_one
!! NAME
!! hubbard_one
!!
!! FUNCTION
!! Solve the hubbard one approximation
!!
!! COPYRIGHT
!! Copyright (C) 1999-2021 ABINIT group (BAmadon)
!! This file is distributed under the terms of the
!! GNU General Public License, see ~abinit/COPYING
!! or http://www.gnu.org/copyleft/gpl.txt .
!! For the initials of contributors, see ~abinit/doc/developers/contributors.txt .
!!
!! INPUTS
!!  cryst_struc
!!  istep    =  step of iteration for DFT.
!!  dft_occup
!!  mpi_enreg=information about MPI parallelization
!!  paw_dmft =  data for self-consistent DFT+DMFT calculations.
!!  pawang <type(pawang)>=paw angular mesh and related data
!!
!! OUTPUT
!!  paw_dmft =  data for self-consistent DFT+DMFT calculations.
!!
!! NOTES
!!
!! PARENTS
!!      m_dmft
!!
!! CHILDREN
!!      combin,destroy_green,init_green,wrtout
!!
!! SOURCE

subroutine hubbard_one(cryst_struc,green,hu,paw_dmft,pawang,pawprtvol,hdc,weiss)

!Arguments ------------------------------------
!scalars
 type(crystal_t),intent(in) :: cryst_struc
 type(green_type), intent(inout) :: green
 type(paw_dmft_type), intent(in)  :: paw_dmft
 type(hu_type), intent(in) :: hu(cryst_struc%ntypat)
 type(pawang_type), intent(in) :: pawang
 type(oper_type), intent(in) :: hdc
 integer, intent(in) :: pawprtvol
 type(green_type), intent(inout) :: weiss

!Local variables ------------------------------
! scalars
 character(len=500) :: message
 integer :: iatom,ifreq,im,im1,isppol,ispinor,ispinor1
 integer :: lpawu,natom,nspinor,nsppol
 
 !----------
!Variables for writing out the NETCDF file
!----------
 integer(kind=4) :: ncid
 integer(kind=4) :: dim_one_id, dim_norb_id, dim_nw_id
 integer(kind=4), dimension(3) :: dim_g_iw_id
 integer(kind=4), dimension(4) :: dim_u_mat_id
 integer(kind=4) :: var_iatom_id, var_nw_id, var_norb_id, var_beta_id
 integer(kind=4) :: var_u_mat_id, var_omega_id 
 integer(kind=4) :: var_real_g0_iw_id, var_imag_g0_iw_id
 integer(kind=4) :: var_real_g_iw_id, var_imag_g_iw_id

 integer :: nw, norb
 real(dp) :: beta
 
 complex(dpc), allocatable :: g0_iw_tmp(:,:,:), g_iw_tmp(:,:,:)

 integer(kind=4) :: varid
 real(dp), allocatable :: real_gimp_tmp(:,:,:), imag_gimp_tmp(:,:,:)
 real(dp), allocatable :: real_g0imp_tmp(:,:,:), imag_g0imp_tmp(:,:,:)
! ************************************************************************

#ifndef HAVE_PYTHON_INVOCATION
 write(message,'(23a)') ch10,' Python invocation flag requiered! You need to install ABINIT with ',&
&  'enable_python_invocation = yes" in your "configure.ac" file.'
 call wrtout(std_out,message,'COLL')
 ABI_ERROR(message)
#endif

 natom=cryst_struc%natom
 nsppol=paw_dmft%nsppol
 nspinor=paw_dmft%nspinor
 if(nsppol/=1.or.nspinor/=1) then
   write(message,'(2a)') " ED not implemented error: nsppol and nspinor should be equal to 1"
   ABI_ERROR(message)
 endif
 isppol=1
 ispinor=1
 ispinor1=1
 
 do iatom=1,cryst_struc%natom
   lpawu=paw_dmft%lpawu(iatom)
   if(lpawu/=-1) then
     ! Creating the NETCDF file 
     nw = paw_dmft%dmft_nwlo
     norb = 2*lpawu+1
     beta = 1.0/paw_dmft%temp
  
     ABI_MALLOC(g0_iw_tmp,(1:norb,1:norb,1:nw))
     do im=1,norb
       do im1=1,norb
         do ifreq=1,nw
           g0_iw_tmp(im,im1,ifreq) = weiss%oper(ifreq)%matlu(iatom)%mat(im,im1,isppol,ispinor,ispinor1)
         end do
       end do
     end do
     
     ABI_MALLOC(g_iw_tmp,(1:norb,1:norb,1:nw))
     do im=1,norb
       do im1=1,norb
         do ifreq=1,nw
           g_iw_tmp(im,im1,ifreq) = green%oper(ifreq)%matlu(iatom)%mat(im,im1,isppol,ispinor,ispinor1)
         end do
       end do
     end do
     
     write(std_out, '(2a)') ch10, "    Creating NETCDF file: test.nc"
     NCF_CHECK(nf90_create("test.nc", NF90_CLOBBER, ncid))
    
     ! Defining the dimensions of the variables to write in the NETCDF file
     NCF_CHECK(nf90_def_dim(ncid, "one", 1, dim_one_id))
     NCF_CHECK(nf90_def_dim(ncid, "norb", norb, dim_norb_id))
     NCF_CHECK(nf90_def_dim(ncid, "nw", nw, dim_nw_id))
    
     dim_u_mat_id = (/ dim_norb_id, dim_norb_id, dim_norb_id, dim_norb_id /)
     dim_g_iw_id = (/ dim_norb_id, dim_norb_id, dim_nw_id /)
   
     ! Defining the variables
     NCF_CHECK(nf90_def_var(ncid, "iatom",           NF90_INT, dim_one_id,           var_iatom_id))
     NCF_CHECK(nf90_def_var(ncid, "nw",              NF90_INT, dim_one_id,           var_nw_id))
     NCF_CHECK(nf90_def_var(ncid, "norb",            NF90_INT, dim_one_id,           var_norb_id))
     NCF_CHECK(nf90_def_var(ncid, "beta",            NF90_FLOAT, dim_one_id,         var_beta_id))
     NCF_CHECK(nf90_def_var(ncid, "u_mat",           NF90_DOUBLE, dim_u_mat_id,      var_u_mat_id))
     NCF_CHECK(nf90_def_var(ncid, "omega",           NF90_DOUBLE, dim_nw_id,         var_omega_id))
     NCF_CHECK(nf90_def_var(ncid, "real_g0_iw",      NF90_DOUBLE, dim_g_iw_id,       var_real_g0_iw_id))
     NCF_CHECK(nf90_def_var(ncid, "imag_g0_iw",      NF90_DOUBLE, dim_g_iw_id,       var_imag_g0_iw_id))
     NCF_CHECK(nf90_def_var(ncid, "real_g_iw",       NF90_DOUBLE, dim_g_iw_id,       var_real_g_iw_id))
     NCF_CHECK(nf90_def_var(ncid, "imag_g_iw",       NF90_DOUBLE, dim_g_iw_id,       var_imag_g_iw_id))
     NCF_CHECK(nf90_enddef(ncid))
     
     ! Filling the variables with actual data
     NCF_CHECK(nf90_put_var(ncid, var_iatom_id,              iatom))
     NCF_CHECK(nf90_put_var(ncid, var_nw_id,                 nw))
     NCF_CHECK(nf90_put_var(ncid, var_norb_id,               norb))
     NCF_CHECK(nf90_put_var(ncid, var_beta_id,               beta))
     NCF_CHECK(nf90_put_var(ncid, var_u_mat_id,              hu(cryst_struc%typat(iatom))%vee))
     NCF_CHECK(nf90_put_var(ncid, var_omega_id,              green%omega))
     NCF_CHECK(nf90_put_var(ncid, var_real_g0_iw_id,         real(g0_iw_tmp)))
     NCF_CHECK(nf90_put_var(ncid, var_imag_g0_iw_id,         aimag(g0_iw_tmp)))
     NCF_CHECK(nf90_put_var(ncid, var_real_g_iw_id,          real(g_iw_tmp)))
     NCF_CHECK(nf90_put_var(ncid, var_imag_g_iw_id,          aimag(g_iw_tmp)))
     NCF_CHECK(nf90_close(ncid))
     call xmpi_barrier(paw_dmft%spacecomm)
   
     write(std_out, '(2a)') ch10, "    NETCDF file test.nc written; Launching python invocation"
     ABI_FREE( g0_iw_tmp )
     ABI_FREE( g_iw_tmp )
    
     ! Invoking python to execute the script
     call Invoke_python_triqs (paw_dmft%myproc, trim(paw_dmft%filnamei)//c_null_char)
   
     write(std_out, '(2a)') ch10, "    Reading NETCDF file test.nc "
   
     ! Opening the NETCDF file
     NCF_CHECK(nf90_open("test.nc", nf90_nowrite, ncid))
   
     ! Read from the file
     ABI_MALLOC(real_gimp_tmp,(1:norb,1:norb,1:nw))
     NCF_CHECK(nf90_inq_varid(ncid, "real_gimp_iw", varid))
     NCF_CHECK(nf90_get_var(ncid, varid, real_gimp_tmp))
     ABI_MALLOC(imag_gimp_tmp,(1:norb,1:norb,1:nw))
     NCF_CHECK(nf90_inq_varid(ncid, "imag_gimp_iw", varid))
     NCF_CHECK(nf90_get_var(ncid, varid, imag_gimp_tmp))
     ABI_MALLOC(real_g0imp_tmp,(1:norb,1:norb,1:nw))
     NCF_CHECK(nf90_inq_varid(ncid, "real_g0imp_iw", varid))
     NCF_CHECK(nf90_get_var(ncid, varid, real_g0imp_tmp))
     ABI_MALLOC(imag_g0imp_tmp,(1:norb,1:norb,1:nw))
     NCF_CHECK(nf90_inq_varid(ncid, "imag_g0imp_iw", varid))
     NCF_CHECK(nf90_get_var(ncid, varid, imag_g0imp_tmp))
     NCF_CHECK(nf90_close(ncid))
     call xmpi_barrier(paw_dmft%spacecomm)
  
     do im=1,norb
       do im1=1,norb
         do ifreq=1,nw
           green%oper(ifreq)%matlu(iatom)%mat(im,im1,isppol,ispinor,ispinor1)=&
&           cmplx(real_gimp_tmp(im,im1,ifreq),imag_gimp_tmp(im,im1,ifreq))
         end do
       end do
     end do
  
     do im=1,norb
       do im1=1,norb
         do ifreq=1,nw
           weiss%oper(ifreq)%matlu(iatom)%mat(im,im1,isppol,ispinor,ispinor1)=&
&           cmplx(real_g0imp_tmp(im,im1,ifreq),imag_g0imp_tmp(im,im1,ifreq))
         end do
       end do
     end do
     
    ABI_FREE( real_gimp_tmp )
    ABI_FREE( imag_gimp_tmp )
    ABI_FREE( real_g0imp_tmp )
    ABI_FREE( imag_g0imp_tmp )
  
  endif
 enddo

end subroutine hubbard_one
!!***

END MODULE m_hubbard_one
!!***
