from django.http import HttpResponse
from django.http import HttpResponseBadRequest
from django.shortcuts import render_to_response
from django.views.generic import TemplateView
from django.views.generic import View

from detection.models import UserDetectionModel
from detection.models import CameraModel

import datetime


class IndexView(TemplateView):
    template_name = 'index.html'

    def get_context_data(self, *args, **kwargs):
        context = super(IndexView, self).get_context_data(**kwargs)
        context['cameras'] = CameraModel.objects.all()
        return context


class DetectionView(View):
    def get(self, *args, **kwargs):
        if UserDetectionModel.objects.filter(user__isnull=True,
                                             datetime__gt=datetime.datetime.now() - datetime.timedelta(
                                                 minutes=5)).exists():
            return HttpResponse()

        return HttpResponseBadRequest()


def daily_detection_chart_view(request):
    knowns = UserDetectionModel.objects.raw(
        """
        SELECT 1 AS id, CAST(strftime('%%H', datetime) as decimal) + 8 AS hour, COUNT(datetime) AS detection
        FROM users_detection
        WHERE user_id IS NOT NULL
        AND strftime('%%Y-%%m-%%d', datetime)  == date('now', 'localtime')
        GROUP BY hour
        """)

    unknowns = UserDetectionModel.objects.raw(
        """
        SELECT 1 AS id, CAST(strftime('%%H', datetime) as decimal) + 8 AS hour, COUNT(datetime) AS detection
        FROM users_detection
        WHERE user_id IS NULL
        AND strftime('%%Y-%%m-%%d', datetime)  == date('now', 'localtime')
        GROUP BY hour
        """)

    x = ['x'] + [r for r in range(0, 24)]
    known_value = ['Detection']
    unknown_value = ['Unknown']

    for r in range(0, 24):
        for known in knowns:
            if r == known.hour:
                known_value.append(known.detection)
                break
        else:
            known_value.append(0)

        for unknown in unknowns:
            if r == unknown.hour:
                unknown_value.append(unknown.detection)
                break
        else:
            unknown_value.append(0)

    return render_to_response('daily-graph.html',
                              {'x_axis': x, 'known_value': known_value, 'unknown_value': unknown_value, })


def monthly_detection_chart_view(request):
    knowns = UserDetectionModel.objects.raw(
        """
        SELECT 1 AS id, CAST(strftime('%%d', datetime) as decimal) AS day, COUNT(datetime) AS detection
        FROM users_detection
        WHERE user_id IS NOT NULL
        AND strftime('%%Y-%%m', datetime)  == strftime('%%Y-%%m', datetime('now', 'localtime'))
        GROUP BY day
        """)

    unknowns = UserDetectionModel.objects.raw(
        """
        SELECT 1 AS id, CAST(strftime('%%d', datetime) as decimal) AS day, COUNT(datetime) AS detection
        FROM users_detection
        WHERE user_id IS NULL
        AND strftime('%%Y-%%m', datetime)  == strftime('%%Y-%%m', datetime('now', 'localtime'))
        GROUP BY day
        """)

    today = datetime.date.today()
    last_day_of_month = datetime.date(today.year, today.month + 1, 1) - datetime.timedelta(days=1)
    x = ['x'] + [r for r in range(1, int(last_day_of_month.day) + 1)]
    known_value = ['Detection']
    unknown_value = ['Unknown']

    for r in range(1, int(last_day_of_month.day) + 1):
        for known in knowns:
            if r == known.day:
                known_value.append(known.detection)
                break
        else:
            known_value.append(0)

        for unknown in unknowns:
            if r == unknown.day:
                unknown_value.append(unknown.detection)
                break
        else:
            unknown_value.append(0)

    return render_to_response('monthly-graph.html',
                              {'x_axis': x, 'known_value': known_value, 'unknown_value': unknown_value, })


def yearly_detection_chart_view(request):
    knowns = UserDetectionModel.objects.raw(
        """
        SELECT 1 AS id, CAST(strftime('%%m', datetime) as decimal) AS month, COUNT(datetime) AS detection
        FROM users_detection
        WHERE user_id IS NOT NULL
        AND strftime('%%Y', datetime)  == strftime('%%Y', datetime('now', 'localtime'))
        GROUP BY month
        """)

    unknowns = UserDetectionModel.objects.raw(
        """
        SELECT 1 AS id, CAST(strftime('%%m', datetime) as decimal) AS month, COUNT(datetime) AS detection
        FROM users_detection
        WHERE user_id IS NULL
        AND strftime('%%Y', datetime)  == strftime('%%Y', datetime('now', 'localtime'))
        GROUP BY month
        """)

    x = ['x'] + [r for r in range(1, 13)]
    known_value = ['Detection']
    unknown_value = ['Unknown']

    for r in range(1, 13):
        for known in knowns:
            if r == known.month:
                known_value.append(known.detection)
                break
        else:
            known_value.append(0)

        for unknown in unknowns:
            if r == unknown.month:
                unknown_value.append(unknown.detection)
                break
        else:
            unknown_value.append(0)

    return render_to_response('yearly-graph.html',
                              {'x_axis': x, 'known_value': known_value, 'unknown_value': unknown_value, })
