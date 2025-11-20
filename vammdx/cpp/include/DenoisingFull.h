/* Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg. */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once

#include <Eigen/Dense>
#include <stdexcept>

#include "Variational.h"
#include "Full.h"

class DenoisingFull
{
public:
    Full &model;
    DenoisingFull(Full &);

    template <class Lmbd>
    void estimate_allocate(const Lmbd& lmbd) const;

    void estimate(cRef<Vector<>>, Ref<Vector<>>, const q_t &) const;

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

    static void bind(pybind11::module_ &m);

#endif
};

//--------------------------------------------------------------------------------------------------------------------//

DenoisingFull::DenoisingFull(Full &_model) : model(_model) {}

template <class Lmbd>
void DenoisingFull::estimate_allocate(const Lmbd& lmbd) const
{    
    lmbd();
}

void DenoisingFull::estimate(cRef<Vector<>>, Ref<Vector<>> x_reco, const q_t &q) const
{
    for (auto &[c, posterior] : q)
    {
        x_reco += posterior * model.M.row(c);
    }
}

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
void DenoisingFull::bind(pybind11::module_ &m)
{
    pybind11::class_<DenoisingFull> DenoisingFull_class_(m, "DenoisingFull", pybind11::module_local());

    DenoisingFull_class_.def(pybind11::init<Full &>(), "model"_a);
}

#endif
