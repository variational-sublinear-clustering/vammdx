/* Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg  */
/* and Artificial Intelligence Lab of the University of Innsbruck.         */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once

#include <Eigen/Dense>
#include <stdexcept>

#include "Variational.h"
#include "MFA.h"

class DenoisingMFA
{
public:
    MFA &model;
    DenoisingMFA(MFA &);

    template <class Lmbd>
    void estimate_allocate(const Lmbd& lmbd) const;

    void estimate(cRef<Vector<>>, Ref<Vector<>>, const q_t &, Vector<>&, ColVector<>&) const;

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

    static void bind(pybind11::module_ &m);

#endif
};

//--------------------------------------------------------------------------------------------------------------------//

DenoisingMFA::DenoisingMFA(MFA &_model) : model(_model) {}

template <class Lmbd>
void DenoisingMFA::estimate_allocate(const Lmbd& lmbd) const
{
    Vector<> T0(model.D);
    ColVector<> T1(model.H);
    
    lmbd(T0, T1);
}

void DenoisingMFA::estimate(cRef<Vector<>> x, Ref<Vector<>> x_reco, const q_t &q, Vector<>& T0, ColVector<>& T1) const
{
    for (auto &[c, posterior] : q)
    {
        T0 = x - model.M.row(c);
        T1.noalias() = T0 * model.UV[c].rightCols(model.H);
        T0.noalias() = model.A.row(c).reshaped<Eigen::RowMajor>(model.D, model.H) * T1;
        x_reco += posterior * (T0 + model.M.row(c));
    }
}

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
void DenoisingMFA::bind(pybind11::module_ &m)
{
    pybind11::class_<DenoisingMFA> DenoisingMFA_class_(m, "DenoisingMFA", pybind11::module_local());

    DenoisingMFA_class_.def(pybind11::init<MFA &>(), "model"_a);
}

#endif
